# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch_xla
from infra.evaluators import ComparisonConfig, EvaluatorFactory
from infra.utilities import (
    Framework,
    Mesh,
    Tensor,
    compile_jax_workload_for_cpu,
    compile_jax_workload_for_tt_device,
    compile_torch_workload_for_cpu,
    compile_torch_workload_for_tt_device,
    random_tensor,
    sanitize_test_name,
)
from infra.workloads import Workload
from infra.workloads.torch_workload import TorchWorkload
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
        torch_options: dict = None,
    ) -> None:
        """Protected constructor for subclasses to use."""
        if compiler_config is None:
            compiler_config = CompilerConfig()
        self._compiler_config = compiler_config
        self._torch_options = torch_options if torch_options is not None else {}
        super().__init__(comparison_config, framework)
        self._initialize_comparison_evaluator()

    def test(self, workload: Workload) -> None:
        """
        Runs test by running `workload` on TT device and CPU and comparing the results.
        """
        cpu_workload = workload
        if self._framework == Framework.JAX:
            compile_jax_workload_for_cpu(cpu_workload)
        else:
            compile_torch_workload_for_cpu(cpu_workload)
        cpu_res = self._device_runner.run_on_cpu(cpu_workload)

        tt_workload = workload
        if self._framework == Framework.JAX:
            compile_jax_workload_for_tt_device(
                tt_workload, self._compiler_config.to_jax_compiler_options()
            )
        else:
            # Must set torch compiler options before compiling for TT device
            if self._compiler_config is not None:
                torch_xla.set_custom_compile_options(
                    self._compiler_config.to_torch_compile_options()
                )
            compile_torch_workload_for_tt_device(tt_workload, self._torch_options)
        tt_res = self._device_runner.run_on_tt_device(tt_workload)

        self._comparison_evaluator.compare(tt_res, cpu_res)

    def _initialize_comparison_evaluator(self) -> None:
        self._comparison_evaluator = EvaluatorFactory.create_evaluator(
            evaluation_type="comparison",
            framework=self._framework,
            comparison_config=self._comparison_config,
        )

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

    def serialize_on_device(self, workload: Workload, output_prefix: str) -> None:
        """
        Serializes a workload on TT device with proper compiler configuration.

        Args:
            workload: The workload to serialize
            output_prefix: Base path and filename prefix for output files
        """
        if self._framework == Framework.JAX:
            compiler_options = self._compiler_config.to_jax_compiler_options()
        elif self._framework == Framework.TORCH:
            compiler_options = self._compiler_config.to_torch_compile_options()
            self._compile_for_tt_device(workload)
        else:
            compiler_options = None

        self._device_runner.serialize_on_device(
            workload, output_prefix, compiler_options=compiler_options
        )


def run_op_test(
    op: Callable,
    inputs: Sequence[Tensor],
    comparison_config: ComparisonConfig = ComparisonConfig(),
    framework: Framework = Framework.JAX,
    compiler_config: CompilerConfig = None,
    mesh: Optional[Mesh] = None,
    shard_spec_fn: Optional[Callable] = None,
) -> None:
    """
    Tests `op` with `inputs` by running it on TT device and CPU and comparing the
    results based on `comparison_config`.
    """
    if compiler_config is None:
        compiler_config = CompilerConfig()
    tester = OpTester(comparison_config, framework, compiler_config=compiler_config)
    if framework == Framework.TORCH:
        workload = TorchWorkload(
            model=op, args=inputs, mesh=mesh, shard_spec_fn=shard_spec_fn
        )
    else:
        workload = Workload(framework, executable=op, args=inputs)
    tester.test(workload)


def serialize_op(
    op: Callable,
    inputs: Sequence[Tensor],
    output_prefix: str,
    framework: Framework = Framework.JAX,
    compiler_config: CompilerConfig = None,
) -> None:
    """
    Serializes an op with given inputs to disk.

    Args:
        op: The operation/function to serialize
        inputs: Input tensors for the operation
        output_prefix: Base path and filename prefix for output files
        framework: The framework to use (default: JAX)
        compiler_config: Compiler configuration options
    """
    # Create an OpTester instance to get access to its device runner
    if compiler_config is None:
        compiler_config = CompilerConfig()
    tester = OpTester(framework=framework, compiler_config=compiler_config)

    workload = Workload(framework=framework, executable=op, args=inputs)

    # Serialize workload on TT device using OpTester's method
    tester.serialize_on_device(workload, output_prefix)


def serialize_op_with_random_inputs(
    op: Callable,
    input_shapes: Sequence[tuple],
    test_name: str,
    minval: float = 0.0,
    maxval: float = 1.0,
    dtype: str | DTypeLike | torch.dtype = "float32",
    framework: Framework = Framework.JAX,
    compiler_config: CompilerConfig = None,
) -> None:
    """
    Serializes an op with random inputs to disk.

    Args:
        op: The operation/function to serialize
        input_shapes: Shapes for random input generation
        test_name: Test name to generate output prefix from
        minval: Minimum value for random inputs (default: 0.0)
        maxval: Maximum value for random inputs (default: 1.0)
        dtype: Data type for inputs
        framework: The framework to use (default: JAX)
        compiler_config: Compiler configuration options
    """

    clean_name = sanitize_test_name(test_name)
    output_prefix = f"output_artifact/{clean_name}"

    inputs = [
        random_tensor(
            shape,
            minval=minval,
            maxval=maxval,
            dtype=dtype,
            framework=framework,
        )
        for shape in input_shapes
    ]
    serialize_op(op, inputs, output_prefix, framework, compiler_config)


def run_op_test_with_random_inputs(
    op: Callable,
    input_shapes: Sequence[tuple],
    minval: float = 0.0,
    maxval: float = 1.0,
    dtype: str | DTypeLike | torch.dtype = "float32",
    comparison_config: ComparisonConfig = ComparisonConfig(),
    framework: Framework = Framework.JAX,
    compiler_config: CompilerConfig = None,
    torch_options: dict = None,
) -> None:
    """
    Tests `op` with random inputs in range [`minval`, `maxval`) by running it on
    TT device and CPU and comparing the results based on `comparison_config`.
    """
    if compiler_config is None:
        compiler_config = CompilerConfig()
    tester = OpTester(
        comparison_config,
        framework,
        compiler_config=compiler_config,
        torch_options=torch_options,
    )
    tester.test_with_random_inputs(op, input_shapes, minval, maxval, dtype)
