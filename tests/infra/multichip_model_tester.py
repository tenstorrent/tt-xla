# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Mapping, Sequence, Union

import jax
from flax import linen
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import PyTree

from .comparison import ComparisonConfig
from .device_connector import device_connector
from .device_runner import DeviceRunner
from .model_tester import ModelTester, RunMode
from .multichip_utils import ShardingMode
from .types import Model
from .workload import MultichipWorkload, Workload


class MultichipModelTester(ModelTester, ABC):
    """
    Abstract base class all multichip model testers must inherit.

    Derived classes must provide implementations of:
    ```
    _get_model
    _get_input_activations_partition_specs
    _get_input_activations
    _get_parameters_partition_specs
    _get_input_parameters # Optional, has default behaviour.
    _get_forward_method_arg_specs # Optional, has default behaviour.
    _get_forward_method_name # Optional, has default behaviour.
    # One of or both:
    _get_forward_method_args # Optional, has default behaviour.
    _get_forward_method_kwargs # Optional, has default behaviour.
    ```
    """

    # ---------- Public methods ----------

    def __init__(
        self,
        mesh_shape: tuple,
        axis_names: tuple,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self.tt_mesh = device_connector.get_tt_device_mesh(mesh_shape, axis_names)
        self.cpu_mesh = device_connector.get_cpu_device_mesh(mesh_shape, axis_names)

        super().__init__(comparison_config, run_mode)

    # ---------- Private methods ----------

    # @override
    def _cache_model_inputs(self) -> None:
        """Caches model inputs."""
        self._input_activations_partition_specs = (
            self._get_input_activations_partition_specs()
        )
        self._input_activations = self._get_input_activations()
        self._input_parameters_partition_specs = (
            self._get_input_parameters_partition_specs()
        )
        self._input_parameters = self._get_input_parameters()

    @abstractmethod
    def _get_input_activations_partition_specs(self) -> PartitionSpec:
        """Returns partition specs for the input activations."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _get_input_parameters_partition_specs(self) -> PyTree:
        """Returns partition specs for the parameters."""
        raise NotImplementedError("Subclasses must implement this method.")

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
    def _test_inference(self) -> None:
        """
        Tests the model by running inference on TT device and on CPU and comparing the
        results.
        """
        ModelTester._configure_model_for_inference(self._model)

        forward_method_arg_specs = self._get_forward_method_arg_specs()

        compiled_tt_workload = MultichipWorkload(
            self._compile_for_device(self.tt_mesh, forward_method_arg_specs),
            self._workload.args,
            self._workload.kwargs,
            device_mesh=self.tt_mesh,
            in_specs=forward_method_arg_specs,
        )
        tt_res = DeviceRunner.run_on_multichip_device(
            compiled_tt_workload, ShardingMode.INPUTS_AND_MODULE
        )

        compiled_cpu_workload = MultichipWorkload(
            self._compile_for_device(self.cpu_mesh, forward_method_arg_specs),
            self._workload.args,
            self._workload.kwargs,
            device_mesh=self.cpu_mesh,
            in_specs=forward_method_arg_specs,
        )
        cpu_res = DeviceRunner.run_on_multichip_device(
            compiled_cpu_workload, ShardingMode.INPUTS_AND_MODULE
        )

        self._compare(tt_res, cpu_res)

    def _compile_for_device(
        self, device_mesh: Mesh, forward_method_arg_specs: tuple[PartitionSpec | PyTree]
    ) -> Callable:
        """
        JIT-compiles model's forward pass into optimized kernels for the given device
        mesh.
        """
        # Assuming replicated outputs for now.
        out_spec = PartitionSpec()
        return jax.jit(
            shard_map(
                self._workload.executable,
                device_mesh,
                in_specs=forward_method_arg_specs,
                out_specs=out_spec,
                # For some reason this check doesn't like replicated outputs.
                check_rep=False,
            ),
            out_shardings=NamedSharding(device_mesh, out_spec),
            static_argnames=self._workload.static_argnames,
        )
