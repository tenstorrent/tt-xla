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
from .device_runner import DeviceRunner
from .multichip_workload import MultichipWorkload
from .workload import Workload


class MultichipTester(BaseTester):
    """Specific tester for ops."""

    def __init__(
        self, mesh: jax.Mesh, in_specs: tuple, out_specs: jax.sharding.PartitionSpec, comparison_config: ComparisonConfig = ComparisonConfig()
    ) -> None:
        self.mesh = mesh
        self.in_specs = in_specs
        self.out_specs = out_specs
        super().__init__(comparison_config)

    def _compile_cpu(
        self, executable: Callable, static_argnames: Sequence[str] = None
    ) -> Callable:
        """Sets up `executable` for just-in-time compile - specifically for CPU."""
        return jax.jit(executable, static_argnames=static_argnames)

    def _compile(
        self, executable: Callable, static_argnames: Sequence[str] = None
    ) -> Callable:
        """Sets up `executable` for just-in-time compile."""
        module_sharded = shard_map(
            executable,
            mesh=self.mesh,
            in_specs=self.in_specs, 
            out_specs=self.out_specs  
        )
        output_sharding = NamedSharding(self.mesh, self.out_specs)
        return jax.jit(module_sharded, out_shardings=output_sharding, static_argnames=static_argnames)

    def test(self, workload: Workload, cpu_workload: Workload) -> None:
        """
        Runs test by running `workload` on TT device and CPU and comparing the results.
        """
        compiled_executable = self._compile(workload.executable)
        cpu_compiled_executable = self._compile_cpu(cpu_workload.executable)

        cpu_compiled_workload = Workload(
            cpu_compiled_executable, cpu_workload.args, cpu_workload.kwargs
        )

        compiled_workload = MultichipWorkload(
            compiled_executable, workload.args, workload.kwargs, mesh = self.mesh, in_specs=self.in_specs
        )

        non_sharded_workload = DeviceRunner.put_with_none_sharding(compiled_workload)

        tt_res = DeviceRunner.run_manual(non_sharded_workload)
        cpu_res = DeviceRunner.run_on_cpu(cpu_compiled_workload)

        self._compare(tt_res, cpu_res)

    def test_with_random_inputs(
        self,
        f: Callable,
        golden_f: Callable,
        input_shapes: Sequence[tuple],
        minval: float = 0.0,
        maxval: float = 1.0,
    ) -> None:
        """
        Tests `f` by running it with random inputs in range [`minval`, `maxval`) on
        TT device and CPU and comparing the results.
        """
        inputs = [
            jax.random.uniform(key = jax.random.key(0), shape = shape, minval=minval, maxval=maxval) for shape in input_shapes
        ]
        workload = Workload(f, inputs)
        cpu_workload = Workload(golden_f, inputs)
        self.test(workload, cpu_workload)


def run_multichip_test_with_random_inputs(
    mesh_test: Callable,
    golden_test: Callable,
    input_shapes: Sequence[tuple],
    mesh: jax.Mesh, 
    in_specs: Sequence[jax.sharding.PartitionSpec], 
    out_specs: jax.sharding.PartitionSpec,
    minval: float = 0.0,
    maxval: float = 1.0,
    comparison_config: ComparisonConfig = ComparisonConfig(),
) -> None:
    """
    Tests `op` with random inputs in range [`minval`, `maxval`) by running it on
    TT device and CPU and comparing the results based on `comparison_config`.
    """
    tester = MultichipTester(mesh, in_specs, out_specs, comparison_config)
    tester.test_with_random_inputs(mesh_test, golden_test, input_shapes, minval, maxval)