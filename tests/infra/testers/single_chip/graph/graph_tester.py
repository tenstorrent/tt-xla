# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
from infra.comparators import ComparisonConfig
from infra.utilities import Framework, Mesh, Tensor
from infra.workloads.torch_workload import TorchWorkload
from infra.workloads.workload import Workload
from jax._src.typing import DTypeLike

from ..op.op_tester import OpTester


class GraphTester(OpTester):
    """
    Specific single chip tester for graphs.

    Currently same as OpTester.
    """

    pass


def run_graph_test(
    graph: Callable,
    inputs: Sequence[Tensor],
    comparison_config: ComparisonConfig = ComparisonConfig(),
    framework: Framework = Framework.JAX,
    mesh: Optional[Mesh] = None,
    shard_spec_fn: Optional[Callable] = None,
) -> None:
    """
    Tests `graph` with `inputs` by running it on TT device and CPU and comparing the
    results based on `comparison_config`.
    """
    tester = GraphTester(comparison_config, framework)
    if framework == Framework.TORCH:
        workload = TorchWorkload(
            model=graph, args=inputs, mesh=mesh, shard_spec_fn=shard_spec_fn
        )
    else:
        workload = Workload(framework=framework, executable=graph, args=inputs)
    tester.test(workload)


def run_graph_test_with_random_inputs(
    graph: Callable,
    input_shapes: Sequence[tuple],
    comparison_config: ComparisonConfig = ComparisonConfig(),
    framework: Framework = Framework.JAX,
    dtype: str | DTypeLike | torch.dtype = "float32",
) -> None:
    """
    Tests `graph` with random inputs by running it on TT device and CPU and comparing
    the results based on `comparison_config`.
    """
    tester = GraphTester(comparison_config, framework)
    tester.test_with_random_inputs(graph, input_shapes, dtype=dtype)
