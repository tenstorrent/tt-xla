# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from infra import run_op_test_with_random_inputs
from utils import Category


# @pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.lax.clamp",
    shlo_op_name="stablehlo.clamp",
)
@pytest.mark.parametrize(
    ["x_shape", "min_shape", "max_shape"],
    [
        [(32, 32), (32, 32), (32, 32)],
        [(64, 64), (64, 64), (64, 64)],
    ],
    ids=lambda val: f"{val}",
)
def test_clamp(x_shape: tuple, min_shape: tuple, max_shape: tuple):
    def clamp(x: jax.Array, min: jax.Array, max: jax.Array) -> jax.Array:
        return jax.lax.clamp(min, x, max)

    run_op_test_with_random_inputs(
        clamp, [x_shape, min_shape, max_shape], minval=-5.0, maxval=5.0
    )
