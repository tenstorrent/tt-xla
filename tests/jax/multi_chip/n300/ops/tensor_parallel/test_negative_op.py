# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import pytest
from infra import (
    make_partition_spec,
    ShardingMode,
    run_multichip_test_with_random_inputs,
)

from tests.utils import failed_fe_compilation


@pytest.mark.nightly
@pytest.mark.push
@pytest.mark.parametrize(
    "use_shardy",
    [
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                reason="Shardy sharding not supported (issue #383)"
            ),
        ),
        False,
    ],
)
@pytest.mark.parametrize(
    ("input_shape", "mesh_shape", "axis_names"), [((256, 256), (1, 2), ("x", "y"))]
)
@pytest.mark.parametrize(
    "sharding_mode",
    [
        ShardingMode.INPUTS_AND_MODULE,
        pytest.param(
            ShardingMode.MODULE,
            marks=pytest.mark.xfail(
                reason=failed_fe_compilation(
                    "Cannot get sharding information through the protobuf "
                    "(https://github.com/tenstorrent/tt-xla/issues/277)"
                )
            ),
        ),
        ShardingMode.INPUTS,
    ],
)
def test_unary_eltwise(
    use_shardy: bool,
    input_shape: tuple,
    mesh_shape: tuple,
    axis_names: tuple,
    sharding_mode: ShardingMode,
):
    def fwd(a_block):
        b_block = jnp.negative(a_block)
        return b_block

    in_specs = (make_partition_spec(axis_names),)
    out_specs = make_partition_spec(axis_names)

    run_multichip_test_with_random_inputs(
        fwd,
        [input_shape],
        mesh_shape,
        axis_names,
        in_specs,
        out_specs,
        use_shardy,
        sharding_mode,
    )
