# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

import jax
from flax import linen
from infra.comparators import ComparisonConfig
from infra.connectors import JaxDeviceConnector
from infra.runners import JaxDeviceRunner
from infra.utilities import PyTree, ShardingMode, Tensor
from infra.workloads import JaxMultichipWorkload, Workload
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec

from tests.infra.testers.compiler_config import CompilerConfig

from ...single_chip import JaxModelTester, RunMode


class JaxMultichipModelTester(JaxModelTester, ABC):
    """
    Abstract base class all multichip `jax` model testers must inherit.

    Derived classes must provide implementations of:
    ```
    _get_model(self) -> Model
    _get_input_activations_partition_spec -> PartitionSpec
    _get_input_activations(self) -> Sequence[Any]
    _get_input_parameters_partition_spec -> PyTree
    _get_input_parameters(self) -> PyTree # Optional, has default behaviour.
    _get_forward_method_name(self) -> str # Optional, has default behaviour.
    _get_forward_method_arg_specs(self) -> tuple[PartitionSpec | PyTree] # Optional, has default behaviour.
    # One of or both:
    _get_forward_method_args(self) -> Sequence[Any] # Optional, has default behaviour.
    _get_forward_method_kwargs(self) -> Mapping[str, Any] # Optional, has default behaviour.
    ```
    """

    def __init__(
        self,
        mesh_shape: tuple,
        axis_names: tuple,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
        compiler_config: CompilerConfig = None,
    ) -> None:
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
    def _test_inference(self) -> None:
        """
        Tests the model by running inference on multichip TT device and on CPU and comparing the
        results.
        """
        self._compile_for_cpu(self.cpu_multichip_workload)
        cpu_res = self._run_on_cpu(self.cpu_multichip_workload)

        self._compile_for_tt_device(self.tt_device_multichip_workload)
        tt_res = self._run_on_tt_device(self.tt_device_multichip_workload)

        self._compare(tt_res, cpu_res)

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

        module_sharded_executable = shard_map(
            workload.executable,
            mesh=workload.device_mesh,
            in_specs=workload.in_specs,
            out_specs=workload.out_spec,
            # For some reason this check doesn't like replicated outputs.
            check_rep=False,
        )
        output_sharding = NamedSharding(workload.device_mesh, workload.out_spec)

        workload.executable = jax.jit(
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
            self._workload.executable,
            self._workload.compiled_executable,
            self._workload.args,
            self._workload.kwargs,
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
        if isinstance(self._model, linen.Module):
            return (
                self._input_parameters_partition_specs,
                self._input_activations_partition_specs,
            )

        return ()

    # @override
    def _run_on_cpu(self, compiled_workload: Workload) -> Tensor:
        """Runs workload on CPU."""
        return self._run_on_multichip_device(self.cpu_multichip_workload)

    # @override
    def _run_on_tt_device(self, compiled_workload: Workload) -> Tensor:
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

    @abstractmethod
    def _get_input_activations_partition_spec(self) -> PartitionSpec:
        """Returns partition specs for the input activations."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _get_input_parameters_partition_spec(self) -> PyTree:
        """Returns partition specs for the parameters."""
        raise NotImplementedError("Subclasses must implement this method.")

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        # We do not handle training yet, so we return empty dict
        return {}

    # @override
    def _get_static_argnames(self) -> Optional[Sequence[str]]:
        # We do not handle training yet, so we return empty list
        return []
