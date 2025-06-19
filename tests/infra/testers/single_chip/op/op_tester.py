# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, Sequence

import jax
import torch
from comparators import ComparisonConfig
from utilities.types import Framework, Tensor
from utilities.utils import random_tensor
from utilities.workloads import Workload, WorkloadFactory
from utilities.workloads.jax_workload import JaxWorkload
from utilities.workloads.torch_workload import TorchWorkload

from ..single_chip_tester import SingleChipTester


class OpTester(SingleChipTester):
    """Specific single chip tester for ops."""

    # -------------------- Public methods --------------------

    def __init__(
        self,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        framework: Framework = Framework.JAX,
    ) -> None:
        super().__init__(comparison_config, framework)

    def test(self, workload: Workload) -> None:
        """
        Runs test by running `workload` on TT device and CPU and comparing the results.
        """
        compiled_workload = self._compile(workload)

        tt_res = self._run_on_tt_device(compiled_workload)
        cpu_res = self._run_on_cpu(compiled_workload)

        self._compare(tt_res, cpu_res)

    def test_with_random_inputs(
        self,
        f: Callable,
        input_shapes: Sequence[tuple],
        minval: float = 0.0,
        maxval: float = 1.0,
    ) -> None:
        """
        Tests `f` by running it with random inputs in range [`minval`, `maxval`) on
        TT device and CPU and comparing the results.
        """
        inputs = [
            random_tensor(
                shape,
                minval=minval,
                maxval=maxval,
                framework=self._framework,
            )
            for shape in input_shapes
        ]
        workload = WorkloadFactory(self._framework).create_workload(
            executable=f, args=inputs
        )
        self.test(workload)

    # -------------------- Private methods --------------------

    # @override
    def _initialize_all_components(self) -> None:
        self._initialize_framework_specific_helpers()

    # @override
    def _compile(self, workload: Workload) -> Workload:
        """
        Compiles executable carried in `workload` based on framework.

        Returns compiled workload.
        """

        def compile_jax_workload(workload: JaxWorkload) -> Workload:
            workload.executable = jax.jit(
                workload.executable, static_argnames=workload.static_argnames
            )
            return workload

        def compile_torch_workload(workload: TorchWorkload) -> Workload:
            assert workload.executable is not None

            workload.executable = torch.compile(workload.executable, backend="openxla")
            return workload

        if self._framework == Framework.JAX:
            assert isinstance(workload, JaxWorkload)
            return compile_jax_workload(workload)
        else:
            assert isinstance(workload, TorchWorkload)
            return compile_torch_workload(workload)


def run_single_chip_op_test(
    op: Callable,
    inputs: Sequence[Tensor],
    comparison_config: ComparisonConfig = ComparisonConfig(),
    framework: Framework = Framework.JAX,
) -> None:
    """
    Tests `op` with `inputs` by running it on TT device and CPU and comparing the
    results based on `comparison_config`.
    """
    tester = OpTester(comparison_config, framework)
    workload = WorkloadFactory(framework).create_workload(executable=op, args=inputs)
    tester.test(workload)


def run_single_chip_op_test_with_random_inputs(
    op: Callable,
    input_shapes: Sequence[tuple],
    minval: float = 0.0,
    maxval: float = 1.0,
    comparison_config: ComparisonConfig = ComparisonConfig(),
    framework: Framework = Framework.JAX,
) -> None:
    """
    Tests `op` with random inputs in range [`minval`, `maxval`) by running it on
    TT device and CPU and comparing the results based on `comparison_config`.
    """
    tester = OpTester(comparison_config, framework)
    tester.test_with_random_inputs(op, input_shapes, minval, maxval)
