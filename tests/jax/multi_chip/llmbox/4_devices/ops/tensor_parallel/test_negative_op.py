# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import pytest
from infra import (
    ShardingMode,
    make_partition_spec,
    run_jax_multichip_op_test_with_random_inputs,
)
from utils import failed_fe_compilation, failed_ttmlir_compilation


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
    ("input_shape", "mesh_shape", "axis_names"), [((256, 256), (1, 4), ("x", "y"))]
)
@pytest.mark.parametrize(
    "multichip_mode",
    [
        pytest.param(
            ShardingMode.INPUTS_AND_MODULE,
            marks=pytest.mark.xfail(
                reason=failed_ttmlir_compilation(
                    "Get device id issue on 4 chips with LLMbox "
                    "(https://github.com/tenstorrent/tt-xla/issues/484)"
                )
            ),
        ),
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
    multichip_mode: ShardingMode,
):
    if multichip_mode == ShardingMode.INPUTS and not use_shardy:
        pytest.xfail(
            "See https://github.com/tenstorrent/tt-xla/issues/3068 for more information"
        )

    def fwd(a_block):
        b_block = jnp.negative(a_block)
        return b_block

    in_specs = (make_partition_spec(axis_names),)
    out_specs = make_partition_spec(axis_names)

    run_jax_multichip_op_test_with_random_inputs(
        fwd,
        [input_shape],
        mesh_shape,
        axis_names,
        in_specs,
        out_specs,
        use_shardy,
        multichip_mode,
    )
