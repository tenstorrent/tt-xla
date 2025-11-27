# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import jax
from flax import linen, nnx
from infra.comparators import ComparisonConfig, ComparisonResult
from infra.connectors import DeviceConnectorFactory, JaxDeviceConnector
from infra.runners import JaxDeviceRunner
from infra.testers.single_chip.model import JaxModelTester, RunMode
from infra.utilities import Framework, PyTree, ShardingMode, Tensor
from infra.workloads import JaxMultichipWorkload, Workload
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec

from tests.infra.testers.compiler_config import CompilerConfig


class DynamicJaxMultiChipModelTester(JaxModelTester):
    """
    Dynamic multichip JAX model tester that works with any ModelLoader from tt_forge_models.

    This class can test any multichip model that follows the ModelLoader pattern,
    eliminating the need for model-specific tester classes.
    """

    def __init__(
        self,
        model_loader=None,
        mesh_shape: tuple = None,
        axis_names: tuple = None,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
        compiler_config: CompilerConfig = None,
        num_devices: Optional[int] = None,
        axis_name: str = "X",
    ) -> None:
        """Initialize the multichip model tester.

        Args:
            model_loader: ModelLoader instance from tt_forge_models. If provided,
                         the tester will use it for all model operations.
            mesh_shape: Shape of the device mesh. Auto-determined if model_loader is provided.
            axis_names: Names of mesh axes. Auto-determined if model_loader is provided.
            comparison_config: Configuration for result comparison.
            run_mode: RunMode.INFERENCE or RunMode.TRAINING.
            compiler_config: Compiler configuration.
            num_devices: Number of devices to use. Auto-detected if None.
            axis_name: Name of the sharding axis (used when model_loader is provided).
        """
        # If model_loader is provided, auto-configure mesh settings
        if model_loader is not None:
            self._model_loader = model_loader

            # Determine number of devices
            if num_devices is None:
                device_connector = DeviceConnectorFactory.create_connector(
                    Framework.JAX
                )
                num_devices = device_connector.get_number_of_tt_devices()

            self.num_devices = num_devices
            self.main_axis_name = axis_name
            mesh_shape = (self.num_devices,)
            axis_names = (self.main_axis_name,)
        else:
            self._model_loader = None
            self.num_devices = None
            self.main_axis_name = None
        self._mesh_shape = mesh_shape
        self._axis_names = axis_names
        # TODO(mrakita): This should be a parameter of model tester, currently only this
        # mode is supported in compiler.
        self._sharding_mode = ShardingMode.INPUTS_AND_MODULE

        self._device_mesh: jax.sharding.Mesh = None
        self._cpu_mesh: jax.sharding.Mesh = None
        self._input_activations_partition_specs: PartitionSpec = None
        self._input_activations: Dict | Sequence[Any] = None
        self._input_parameters_partition_specs: PyTree = None
        self._input_parameters: PyTree = None

        super().__init__(comparison_config, run_mode, compiler_config)

    # @override
    def _initialize_components(self) -> None:
        self._initialize_meshes()
        super()._initialize_components()

    # @override
    def _initialize_workload(self) -> None:
        super()._initialize_workload()
        # then replace single chip workload with multichip one
        self.tt_device_multichip_workload = self._create_multichip_workload(
            self._device_mesh
        )
        self.cpu_multichip_workload = self._create_multichip_workload(self._cpu_mesh)

    def _initialize_meshes(self) -> None:
        """Initializes `self._device_mesh` and `self._cpu_mesh`."""
        self._device_mesh = self._get_tt_device_mesh()
        self._cpu_mesh = self._get_cpu_device_mesh()

    def _get_tt_device_mesh(self) -> jax.sharding.Mesh:
        """Returns TT device mesh with specified `shape` and `axis_names`."""
        assert isinstance(self._device_runner, JaxDeviceRunner) and isinstance(
            self._device_runner.connector, JaxDeviceConnector
        )
        return self._device_runner.connector.get_tt_device_mesh(
            self._mesh_shape, self._axis_names
        )

    def _get_cpu_device_mesh(self) -> jax.sharding.Mesh:
        """Returns CPU mesh with specified `shape` and `axis_names`."""
        assert isinstance(self._device_runner, JaxDeviceRunner) and isinstance(
            self._device_runner.connector, JaxDeviceConnector
        )
        return self._device_runner.connector.get_cpu_device_mesh(
            self._mesh_shape, self._axis_names
        )

    # @override
    def _test_inference(self) -> Tuple[ComparisonResult, ...]:
        """
        Tests the model by running inference on multichip TT device and on CPU and comparing the
        results.
        """
        self._compile_for_cpu(self.cpu_multichip_workload)
        cpu_res = self._run_on_cpu()

        self._compile_for_tt_device(self.tt_device_multichip_workload)
        tt_res = self._run_on_tt_device()

        return (self._compare(tt_res, cpu_res),)

    # @override
    def _compile_for_cpu(self, workload: Workload) -> None:
        # Compile options are not used for CPU compilation since they are TT backend specific.
        self._compile(workload, compiler_options={})

    # @override
    def _compile_for_tt_device(self, workload: Workload) -> None:
        compiler_options = self._compiler_config.to_jax_compiler_options()
        self._compile(workload, compiler_options)

    # @override
    def _compile(self, workload: Workload, compiler_options: Dict[str, str]) -> None:
        """Compiles `workload` for TT device.

        Sets up `workload.executable` for just-in-time compile and execution.
        `workload.device_mesh` defines for which device (TT or CPU) it will be compiled.
        """
        assert isinstance(workload, JaxMultichipWorkload)

        if not isinstance(workload.executable, nnx.Module):
            module_sharded_executable = shard_map(
                workload.executable,
                mesh=workload.device_mesh,
                in_specs=workload.in_specs,
                out_specs=workload.out_spec,
                # For some reason this check doesn't like replicated outputs.
                check_rep=False,
            )
        else:
            module_sharded_executable = workload.executable

        output_sharding = NamedSharding(workload.device_mesh, workload.out_spec)

        workload.compiled_executable = jax.jit(
            module_sharded_executable,
            out_shardings=output_sharding,
            static_argnames=workload.static_argnames,
            compiler_options=compiler_options,
        )

    def _create_multichip_workload(
        self, mesh: jax.sharding.Mesh
    ) -> JaxMultichipWorkload:
        """
        Creates multichip workload from single chip workload created during class object
        setup and provided `mesh`.
        """
        assert (
            self._workload.is_jax
        ), "Workload must be JAX workload to create JAX multichip workload"

        in_specs = self._get_forward_method_arg_specs()
        out_spec = PartitionSpec()  # Assuming replicated outputs for now.

        return JaxMultichipWorkload(
            executable=self._workload.executable,
            compiled_executable=self._workload.compiled_executable,
            model=self._model,
            args=self._workload.args,
            kwargs=self._workload.kwargs,
            device_mesh=mesh,
            in_specs=in_specs,
            out_spec=out_spec,
            sharding_mode=self._sharding_mode,
        )

    def _get_forward_method_arg_specs(self) -> tuple[PartitionSpec | PyTree]:
        """
        Returns partition specs for the forward method arguments.

        By default returns specs for input parameters and activations for the Flax linen
        models, and empty tuple for other type of models.
        """
        if isinstance(self._model, (linen.Module, nnx.Module)):
            return (
                self._input_parameters_partition_specs,
                self._input_activations_partition_specs,
            )

        return ()

    # @override
    def _run_on_cpu(self) -> Tensor:
        """Runs workload on CPU."""
        return self._run_on_multichip_device(self.cpu_multichip_workload)

    # @override
    def _run_on_tt_device(self) -> Tensor:
        """Runs workload on TT device."""
        return self._run_on_multichip_device(self.tt_device_multichip_workload)

    def _run_on_multichip_device(self, compiled_workload: Workload) -> Tensor:
        """Runs multichip workload on a multichip device."""
        assert isinstance(self._device_runner, JaxDeviceRunner)
        return self._device_runner.run_on_multichip_device(compiled_workload)

    def _get_number_of_tt_devices(self) -> int:
        """Returns number of available TT devices."""
        assert isinstance(self._device_runner, JaxDeviceRunner) and isinstance(
            self._device_runner.connector, JaxDeviceConnector
        )
        return self._device_runner.connector.get_number_of_tt_devices()

    # @override
    def _cache_model_inputs(self) -> None:
        """Caches model inputs."""
        self._input_activations_partition_specs = (
            self._get_input_activations_partition_spec()
        )
        self._input_activations = self._get_input_activations()
        self._input_parameters_partition_specs = (
            self._get_input_parameters_partition_spec()
        )
        self._input_parameters = self._get_input_parameters()

    # Override abstract methods to use model_loader when available
    def _get_model(self):
        """Get the model instance."""
        if self._model_loader is not None:
            if hasattr(self._model_loader, "load_multichip_model"):
                return self._model_loader.load_multichip_model(
                    axis_name=self.main_axis_name,
                    num_devices=self.num_devices,
                    train_mode=self._run_mode == RunMode.TRAINING,
                )
            elif hasattr(self._model_loader, "load_model"):
                return self._model_loader.load_model()
            else:
                raise NotImplementedError(
                    "Model loader must have either load_model or load_multichip_model method"
                )
        else:
            raise NotImplementedError(
                "Must provide model_loader or override _get_model"
            )

    def _get_forward_method_name(self) -> str:
        """Get the forward method name."""
        # This is solo used for Linen models (AlexNet and MNIST)
        return "apply"

    def _get_input_activations(self):
        """Get input activations."""
        if self._model_loader is not None:
            return self._model_loader.load_inputs(mesh=self._cpu_mesh)
        else:
            raise NotImplementedError(
                "Must provide model_loader or override _get_input_activations"
            )

    def _get_input_activations_partition_spec(self) -> PartitionSpec:
        """Returns partition specs for the input activations."""
        if self._model_loader is not None:
            return self._model_loader.get_input_activations_partition_spec(
                self._cpu_mesh, axis_name=self.main_axis_name
            )
        else:
            raise NotImplementedError(
                "Must provide model_loader or override _get_input_activations_partition_spec"
            )

    def _get_input_parameters_partition_spec(self) -> PyTree:
        """Returns partition specs for the parameters."""
        if self._model_loader is not None:
            return self._model_loader.load_parameters_partition_spec(
                model_for_multichip=self._model,
                cpu_mesh=self._cpu_mesh,
                input_activations_partition_specs=self._input_activations_partition_specs,
                inputs=self._input_activations,
            )
        else:
            raise NotImplementedError(
                "Must provide model_loader or override _get_input_parameters_partition_spec"
            )

    def _get_input_parameters(self) -> PyTree:
        """Returns the input parameters."""
        if self._model_loader is not None and hasattr(
            self._model_loader, "load_parameters"
        ):
            return self._model_loader.load_parameters(
                train=self._run_mode == RunMode.TRAINING,
                inputs=self._input_activations,
                model_for_multichip=self._model,
                cpu_mesh=self._cpu_mesh,
                input_activations_partition_specs=self._input_activations_partition_specs,
                input_parameters_partition_specs=self._input_parameters_partition_specs,
            )
        else:
            return super()._get_input_parameters()

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        return {}

    # @override
    def _get_static_argnames(self) -> Optional[Sequence[str]]:
        return []
