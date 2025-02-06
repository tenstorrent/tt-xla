# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
from jax.sharding import NamedSharding, PartitionSpec
from jax.experimental.shard_map import shard_map
from typing import Callable, Sequence

from .base_tester import BaseTester
from .comparison import ComparisonConfig
from .device_runner import DeviceRunner, device_connector
from .workload import MultichipWorkload
from .workload import Workload


class MultichipTester(BaseTester):
    """
    A tester for evaluating operations in a multichip JAX execution environment.

    This class extends `BaseTester` and provides functionality for testing
    operations using a specified device mesh, input sharding specifications,
    and output sharding specifications.

    Attributes:
        mesh (jax.Mesh): The device mesh over which the computation is distributed.
        in_specs (tuple): The sharding specifications for the input tensors.
        out_specs (jax.sharding.PartitionSpec): The sharding specification for the output tensor.
    """

    def __init__(
        self,
        mesh: jax.Mesh,
        in_specs: tuple,
        out_specs: jax.sharding.PartitionSpec,
        comparison_config: ComparisonConfig = ComparisonConfig(),
    ) -> None:
        self.mesh = mesh
        self.in_specs = in_specs
        self.out_specs = out_specs
        super().__init__(comparison_config)

    def _compile_for_cpu(
        self, executable: Callable, static_argnames: Sequence[str] = None
    ) -> Callable:
        """Sets up `executable` for just-in-time compile and execution on CPU"""
        return jax.jit(executable, static_argnames=static_argnames)

    def _compile_for_device(
        self, executable: Callable, static_argnames: Sequence[str] = None
    ) -> Callable:
        """Sets up executable for just-in-time compile and execution on multichip device."""
        module_sharded = shard_map(
            executable, mesh=self.mesh, in_specs=self.in_specs, out_specs=self.out_specs
        )
        output_sharding = NamedSharding(self.mesh, self.out_specs)
        return jax.jit(
            module_sharded,
            out_shardings=output_sharding,
            static_argnames=static_argnames,
        )

    def test(
        self, multichip_workload: MultichipWorkload, cpu_workload: Workload
    ) -> None:
        """
        Runs test by running `workload` on TT device and CPU and comparing the results.
        """
        multichip_compiled_workload = MultichipWorkload(
            self._compile_for_device(multichip_workload.executable),
            multichip_workload.args,
            multichip_workload.kwargs,
            mesh=self.mesh,
            in_specs=self.in_specs,
        )

        cpu_compiled_workload = Workload(
            self._compile_for_cpu(cpu_workload.executable),
            cpu_workload.args,
            cpu_workload.kwargs,
        )

        tt_multichip_res = DeviceRunner.run_on_multichip_device(
            multichip_compiled_workload
        )
        cpu_res = DeviceRunner.run_on_cpu(cpu_compiled_workload)

        self._compare(tt_multichip_res, cpu_res)

    def test_with_random_inputs(
        self,
        device_executable: Callable,
        cpu_executable: Callable,
        input_shapes: Sequence[tuple],
        minval: float = 0.0,
        maxval: float = 1.0,
    ) -> None:
        """
        Tests an input executable with random inputs in range [`minval`, `maxval`) by running it on a mesh of
        TT devices and comparing it to output of the cpu executable ran with the same input.
        """
        inputs = [
            jax.random.uniform(
                key=jax.random.key(0), shape=shape, minval=minval, maxval=maxval
            )
            for shape in input_shapes
        ]
        multichip_workload = MultichipWorkload(
            device_executable, inputs, mesh=self.mesh, in_specs=self.in_specs
        )
        cpu_workload = Workload(cpu_executable, inputs)
        self.test(multichip_workload, cpu_workload)


def run_multichip_test_with_random_inputs(
    device_executable: Callable,
    cpu_executable: Callable,
    input_shapes: Sequence[tuple],
    mesh_shape: tuple,
    axis_names: tuple,
    in_specs: Sequence[jax.sharding.PartitionSpec],
    out_specs: jax.sharding.PartitionSpec,
    minval: float = 0.0,
    maxval: float = 1.0,
    comparison_config: ComparisonConfig = ComparisonConfig(),
) -> None:
    """
    Tests an input executable with random inputs in range [`minval`, `maxval`) by running it on a mesh of
    TT devices and comparing it to output of the cpu executable ran with the same input.
    """
    mesh = device_connector.get_tt_device_mesh(mesh_shape, axis_names)
    tester = MultichipTester(mesh, in_specs, out_specs, comparison_config)
    tester.test_with_random_inputs(
        device_executable, cpu_executable, input_shapes, minval, maxval
    )
