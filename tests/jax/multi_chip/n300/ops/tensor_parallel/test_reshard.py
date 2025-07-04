# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from infra import (
    ShardingMode,
    make_partition_spec,
    run_jax_multichip_op_test_with_random_inputs,
)
from utils import failed_ttmlir_compilation


@pytest.mark.nightly
@pytest.mark.push
@pytest.mark.parametrize(
    "use_shardy",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    ("input_shape", "mesh_shape", "axis_names"), [((32, 32), (1, 2), ("x", "y"))]
)
@pytest.mark.parametrize(
    "sharding_mode",
    [
        ShardingMode.INPUTS,
        ShardingMode.INPUTS_AND_MODULE,
    ],
)
def test_reshard(
    use_shardy: bool,
    input_shape: tuple,
    mesh_shape: tuple,
    axis_names: tuple,
    sharding_mode: ShardingMode,
):
    compiler_options = {}
    if use_shardy:
      compiler_options = {
        "automatic_parallelization": "true",
        "mesh_shape": "1,8"
      }

    def fwd(a_block):
        b_block = jax.lax.with_sharding_constraint(
            a_block, make_partition_spec([None, None])
        )
        return b_block

    in_specs = (make_partition_spec(axis_names),)
    out_specs = make_partition_spec([axis_names[1], axis_names[0]])

    run_jax_multichip_op_test_with_random_inputs(
        fwd,
        [input_shape],
        mesh_shape,
        axis_names,
        in_specs,
        out_specs,
        use_shardy,
        sharding_mode,
        compiler_options=compiler_options
    )
