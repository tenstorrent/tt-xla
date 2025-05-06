# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
import jax
from jax.sharding import NamedSharding
from jax.experimental.shard_map import shard_map
from typing import Callable, Sequence

from .base_tester import BaseTester
from .comparison import ComparisonConfig
from .device_runner import DeviceRunner, device_connector
from .multichip_utils import enable_shardy, ShardingMode
from .workload import MultichipWorkload
from .workload import Workload


class MultichipTester(BaseTester):
    """
    A tester for evaluating operations in a multichip JAX execution environment.

    This class extends `BaseTester` and provides functionality for testing
    operations using a specified device mesh, input sharding specifications,
    and output sharding specifications.

    Attributes:
        device_mesh (jax.Mesh): The device mesh over which the computation is distributed.
        in_specs (tuple): The sharding specifications for the input tensors.
        out_specs (jax.sharding.PartitionSpec): The sharding specification for the output tensor.
    """

    # ---------- Public methods ----------

    def __init__(
        self,
        in_specs: tuple[jax.sharding.PartitionSpec],
        out_specs: jax.sharding.PartitionSpec,
        mesh_shape: tuple,
        axis_names: tuple,
        comparison_config: ComparisonConfig = ComparisonConfig(),
    ) -> None:
        super().__init__(comparison_config)
        self.in_specs = in_specs
        self.out_specs = out_specs
        self.device_mesh = device_connector.get_tt_device_mesh(mesh_shape, axis_names)
        self.cpu_mesh = device_connector.get_cpu_device_mesh(mesh_shape, axis_names)

    def test(
        self,
        multichip_workload: MultichipWorkload,
        cpu_workload: Workload,
        sharding_mode: ShardingMode,
    ) -> None:
        """
        Runs test by running `workload` on TT device and 'cpu_workload' on the CPU and comparing the results.
        """
        with self.device_mesh:
            compiled_device_workload = MultichipWorkload(
                self._compile_for_device(multichip_workload.executable, sharding_mode),
                multichip_workload.args,
                multichip_workload.kwargs,
                device_mesh=self.device_mesh,
                in_specs=self.in_specs,
            )
            device_res = DeviceRunner.run_on_multichip_device(
                compiled_device_workload, sharding_mode
            )

        with self.cpu_mesh:
            compiled_cpu_workload = MultichipWorkload(
                self._compile_for_cpu(cpu_workload.executable, sharding_mode),
                cpu_workload.args,
                cpu_workload.kwargs,
                device_mesh=self.cpu_mesh,
                in_specs=self.in_specs,
            )

            cpu_res = DeviceRunner.run_on_multichip_device(
                compiled_cpu_workload, sharding_mode
            )

        self._compare(device_res, cpu_res)

    def test_with_random_inputs(
        self,
        executable: Callable,
        input_shapes: Sequence[tuple],
        sharding_mode: ShardingMode,
        minval: float = 0.0,
        maxval: float = 1.0,
    ) -> None:
        """
        Tests an input executable with random inputs in range [`minval`, `maxval`) by running it on
        a mesh of TT devices and comparing it to output of the cpu executable ran with the same
        input.
        """
        inputs = [
            jax.random.uniform(
                key=jax.random.key(0), shape=shape, minval=minval, maxval=maxval
            )
            for shape in input_shapes
        ]
        device_workload = MultichipWorkload(
            executable,
            inputs,
            device_mesh=self.device_mesh,
            in_specs=self.in_specs,
        )
        cpu_workload = MultichipWorkload(
            executable,
            inputs,
            device_mesh=self.cpu_mesh,
            in_specs=self.in_specs,
        )

        self.test(device_workload, cpu_workload, sharding_mode)

    # ---------- Private methods ----------

    def _compile_for_cpu(
        self,
        executable: Callable,
        sharding_mode: ShardingMode,
        static_argnames: Sequence[str] = None,
    ) -> Callable:
        """Sets up `executable` for just-in-time compile and execution on CPU"""
        module_sharded = (
            shard_map(
                executable,
                mesh=self.cpu_mesh,
                in_specs=self.in_specs,
                out_specs=self.out_specs,
            )
            if sharding_mode.requires_shard_map
            else executable
        )
        output_sharding = NamedSharding(self.cpu_mesh, self.out_specs)
        return jax.jit(
            module_sharded,
            out_shardings=output_sharding,
            static_argnames=static_argnames,
        )

    def _compile_for_device(
        self,
        executable: Callable,
        sharding_mode: ShardingMode,
        static_argnames: Sequence[str] = None,
    ) -> Callable:
        """Sets up executable for just-in-time compile and execution on multichip device."""
        module_sharded = (
            shard_map(
                executable,
                mesh=self.device_mesh,
                in_specs=self.in_specs,
                out_specs=self.out_specs,
            )
            if sharding_mode.requires_shard_map
            else executable
        )
        output_sharding = NamedSharding(self.device_mesh, self.out_specs)
        return jax.jit(
            module_sharded,
            out_shardings=output_sharding,
            static_argnames=static_argnames,
        )


def run_multichip_test_with_random_inputs(
    executable: Callable,
    input_shapes: Sequence[tuple],
    mesh_shape: tuple,
    axis_names: tuple,
    in_specs: Sequence[jax.sharding.PartitionSpec],
    out_specs: jax.sharding.PartitionSpec,
    use_shardy: bool,
    sharding_mode: ShardingMode,
    minval: float = 0.0,
    maxval: float = 1.0,
    comparison_config: ComparisonConfig = ComparisonConfig(),
) -> None:
    """
    Tests an input executable with random inputs in range [`minval`, `maxval`) by running it on a
    mesh of TT devices and comparing it to output of the cpu executable ran with the same input.
    The xla backend used the shardy dialect if `use_shardy` is True, otherwise it uses GSPMD.
    """
    with enable_shardy(use_shardy), device_connector.simulate_cpu_mesh(mesh_shape):
        tester = MultichipTester(
            in_specs, out_specs, mesh_shape, axis_names, comparison_config
        )
        tester.test_with_random_inputs(
            executable, input_shapes, sharding_mode, minval, maxval
        )
