# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, Sequence

from comparators import ComparisonConfig
from utilities.types import Framework, Tensor
from utilities.workloads import WorkloadFactory

from .op_tester import OpTester


class GraphTester(OpTester):
    """
    Specific tester for graphs.

    Currently same as OpTester.
    """

    pass


def run_graph_test(
    graph: Callable,
    inputs: Sequence[Tensor],
    comparison_config: ComparisonConfig = ComparisonConfig(),
    framework: Framework = Framework.JAX,
) -> None:
    """
    Tests `graph` with `inputs` by running it on TT device and CPU and comparing the
    results based on `comparison_config`.
    """
    tester = GraphTester(comparison_config, framework)
    workload = WorkloadFactory(framework).create_workload(executable=graph, args=inputs)
    tester.test(workload)


def run_graph_test_with_random_inputs(
    graph: Callable,
    input_shapes: Sequence[tuple],
    comparison_config: ComparisonConfig = ComparisonConfig(),
    framework: Framework = Framework.JAX,
) -> None:
    """
    Tests `graph` with random inputs by running it on TT device and CPU and comparing
    the results based on `comparison_config`.
    """
    tester = GraphTester(comparison_config, framework)
    tester.test_with_random_inputs(graph, input_shapes)
