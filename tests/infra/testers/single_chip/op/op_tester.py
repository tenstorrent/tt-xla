# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, Sequence

import jax
import torch
import torch_xla
from infra.comparators import ComparisonConfig
from infra.utilities import Framework, Tensor, random_tensor
from infra.workloads import Workload
from jax._src.typing import DTypeLike

from tests.infra.testers.compiler_config import CompilerConfig

from ...base_tester import BaseTester


class OpTester(BaseTester):
    """Specific single chip tester for ops."""

    def __init__(
        self,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        framework: Framework = Framework.JAX,
        compiler_config: CompilerConfig = None,
    ) -> None:
        """Protected constructor for subclasses to use."""
        if compiler_config is None:
            compiler_config = CompilerConfig()
        self._compiler_config = compiler_config
        super().__init__(comparison_config, framework)

    def test(self, workload: Workload) -> None:
        """
        Runs test by running `workload` on TT device and CPU and comparing the results.
        """
        cpu_workload = workload
        self._compile_for_cpu(cpu_workload)
        cpu_res = self._device_runner.run_on_cpu(cpu_workload)

        tt_workload = workload
        self._compile_for_tt_device(tt_workload)
        tt_res = self._device_runner.run_on_tt_device(tt_workload)

        self._comparator.compare(tt_res, cpu_res)

    def _compile_for_tt_device(self, workload: Workload) -> None:
        """
        Compiles executable carried in `workload` based on framework.
        """

        def compile_jax_workload(workload: Workload) -> None:
            compiler_options = self._compiler_config.to_jax_compiler_options()
            workload.compiled_executable = jax.jit(
                workload.executable,
                static_argnames=workload.static_argnames,
                compiler_options=compiler_options,
            )

        def compile_torch_workload(workload: Workload) -> None:
            assert (workload.executable is None) != (workload.model is None)

            to_compile = (
                workload.model if workload.model is not None else workload.executable
            )
            # Set custom compile options if provided.
            # Use explicit API for passing compiler options.
            if self._compiler_config is not None:
                torch_xla.set_custom_compile_options(
                    self._compiler_config.to_torch_compile_options()
                )
            workload.compiled_executable = torch.compile(to_compile, backend="tt")

        if self._framework == Framework.JAX:
            assert workload.is_jax, "Workload must be JAX workload to compile"
            compile_jax_workload(workload)
        else:
            assert workload.is_torch, "Workload must be Torch workload to compile"
            compile_torch_workload(workload)

    def _compile_for_cpu(self, workload: Workload) -> None:
        """
        Compiles executable carried in `workload` for CPU based on framework.
        """

        def compile_jax_workload(workload: Workload) -> None:
            workload.compiled_executable = jax.jit(
                workload.executable,
                static_argnames=workload.static_argnames,
            )

        def compile_torch_workload(workload: Workload) -> None:
            assert (workload.executable is None) != (workload.model is None)

            to_compile = (
                workload.model if workload.model is not None else workload.executable
            )
            workload.compiled_executable = torch.compile(to_compile, backend="inductor")

        if self._framework == Framework.JAX:
            assert workload.is_jax, "Workload must be JAX workload to compile"
            compile_jax_workload(workload)
        else:
            assert workload.is_torch, "Workload must be Torch workload to compile"
            compile_torch_workload(workload)

    def test_with_random_inputs(
        self,
        f: Callable,
        input_shapes: Sequence[tuple],
        minval: float = 0.0,
        maxval: float = 1.0,
        dtype: str | DTypeLike | torch.dtype = "float32",
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
                dtype=dtype,
                framework=self._framework,
            )
            for shape in input_shapes
        ]
        workload = Workload(framework=self._framework, executable=f, args=inputs)
        self.test(workload)


def run_op_test(
    op: Callable,
    inputs: Sequence[Tensor],
    comparison_config: ComparisonConfig = ComparisonConfig(),
    framework: Framework = Framework.JAX,
    compiler_config: CompilerConfig = None,
) -> None:
    """
    Tests `op` with `inputs` by running it on TT device and CPU and comparing the
    results based on `comparison_config`.
    """
    if compiler_config is None:
        compiler_config = CompilerConfig()
    tester = OpTester(comparison_config, framework, compiler_config=compiler_config)
    workload = Workload(framework, executable=op, args=inputs)
    tester.test(workload)


def run_op_test_with_random_inputs(
    op: Callable,
    input_shapes: Sequence[tuple],
    minval: float = 0.0,
    maxval: float = 1.0,
    dtype: str | DTypeLike | torch.dtype = "float32",
    comparison_config: ComparisonConfig = ComparisonConfig(),
    framework: Framework = Framework.JAX,
    compiler_config: CompilerConfig = None,
) -> None:
    """
    Tests `op` with random inputs in range [`minval`, `maxval`) by running it on
    TT device and CPU and comparing the results based on `comparison_config`.
    """
    if compiler_config is None:
        compiler_config = CompilerConfig()
    tester = OpTester(comparison_config, framework, compiler_config=compiler_config)
    tester.test_with_random_inputs(op, input_shapes, minval, maxval, dtype)
