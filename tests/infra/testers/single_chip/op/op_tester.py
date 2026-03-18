# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
from infra.evaluators import ComparisonConfig
from infra.utilities import Framework, Mesh, Tensor
from infra.workloads import Workload
from infra.workloads.torch_workload import TorchWorkload
from jax._src.typing import DTypeLike

from tests.infra.testers.compiler_config import CompilerConfig
from tests.infra.testers.tester import Tester

# Keep OpTester as an alias for backward compatibility during migration.
# It is functionally identical to Tester.
OpTester = Tester


def run_op_test(
    op: Callable,
    inputs: Sequence[Tensor],
    comparison_config: ComparisonConfig = ComparisonConfig(),
    framework: Framework = Framework.JAX,
    compiler_config: CompilerConfig = None,
    mesh: Optional[Mesh] = None,
    shard_spec_fn: Optional[Callable] = None,
    request=None,
) -> None:
    """
    Tests `op` with `inputs` by running it on TT device and CPU and comparing the
    results based on `comparison_config`.
    """
    if compiler_config is None:
        compiler_config = CompilerConfig()
    tester = Tester(framework, comparison_config, compiler_config=compiler_config)
    if framework == Framework.TORCH:
        workload = TorchWorkload(
            model=op, args=inputs, mesh=mesh, shard_spec_fn=shard_spec_fn
        )
    else:
        workload = Workload(framework, executable=op, args=inputs)
    tester.test(workload, request=request)


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
    request=None,
) -> None:
    """
    Tests `op` with random inputs in range [`minval`, `maxval`) by running it on
    TT device and CPU and comparing the results based on `comparison_config`.
    """
    if compiler_config is None:
        compiler_config = CompilerConfig()
    tester = Tester(
        framework,
        comparison_config,
        compiler_config=compiler_config,
        torch_options=torch_options,
    )
    tester.test_with_random_inputs(
        op, input_shapes, minval, maxval, dtype, request=request
    )
