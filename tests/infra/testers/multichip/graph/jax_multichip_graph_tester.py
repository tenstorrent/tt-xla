# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, Sequence

import jax
from infra.evaluators import ComparisonConfig
from infra.utilities import ShardingMode, enable_shardy

from ..op.jax_multichip_op_tester import JaxMultichipOpTester


class JaxMultichipGraphTester(JaxMultichipOpTester):
    """
    Specific multichip tester for graphs.

    Currently same as JaxMultichipOpTester.
    """

    pass


def run_jax_multichip_graph_test_with_random_inputs(
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
    Tests an input executable with random inputs in range [`minval`, `maxval`) by
    running it on a mesh of TT devices and comparing it to output of the cpu executable
    ran with the same input. The xla backend used the shardy dialect if `use_shardy` is
    True, otherwise it uses GSPMD.
    """
    with enable_shardy(use_shardy):
        tester = JaxMultichipGraphTester(
            in_specs, out_specs, mesh_shape, axis_names, comparison_config
        )
        tester.test_with_random_inputs(
            executable, input_shapes, sharding_mode, minval, maxval
        )
