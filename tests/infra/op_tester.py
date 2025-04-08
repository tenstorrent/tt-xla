# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, Sequence

from .base_tester import BaseTester
from .comparison import ComparisonConfig
from .device_runner import DeviceRunner
from .types import Tensor
from .utils import random_tensor
from .workload import Workload
from .interpreter_check import InterpreterCheck


class OpTester(BaseTester):
    """Specific tester for ops."""

    def __init__(
        self, comparison_config: ComparisonConfig = ComparisonConfig()
    ) -> None:
        super().__init__(comparison_config)

    def test(self, workload: Workload) -> None:
        """
        Runs test by running `workload` on TT device and CPU and comparing the results.
        """
        compiled_executable = self._compile(workload.executable)

        compiled_workload = Workload(
            compiled_executable, workload.args, workload.kwargs
        )

        tt_res = DeviceRunner.run_on_tt_device(compiled_workload)
        cpu_res = DeviceRunner.run_on_cpu(compiled_workload)

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
            random_tensor(shape, minval=minval, maxval=maxval) for shape in input_shapes
        ]
        workload = Workload(f, inputs)
        self.test(workload)


def run_op_test(
    op: Callable,
    inputs: Sequence[Tensor],
    comparison_config: ComparisonConfig = ComparisonConfig(),
) -> None:
    """
    Tests `op` with `inputs` by running it on TT device and CPU and comparing the
    results based on `comparison_config`.
    """
    tester = OpTester(comparison_config)
    workload = Workload(op, inputs)
    tester.test(workload)
    int_check = InterpreterCheck("interpreter_log")
    int_check.compare_tensors()


def run_op_test_with_random_inputs(
    op: Callable,
    input_shapes: Sequence[tuple],
    minval: float = 0.0,
    maxval: float = 1.0,
    comparison_config: ComparisonConfig = ComparisonConfig(),
) -> None:
    """
    Tests `op` with random inputs in range [`minval`, `maxval`) by running it on
    TT device and CPU and comparing the results based on `comparison_config`.
    """
    tester = OpTester(comparison_config)
    tester.test_with_random_inputs(op, input_shapes, minval, maxval)
    int_check = InterpreterCheck("interpreter_log")
    int_check.compare_tensors()
