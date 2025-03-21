# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.lax as jlx
import pytest
from infra import run_op_test_with_random_inputs

from tests.utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.lax.rem",
    shlo_op_name="stablehlo.remainder",
)
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
    ids=lambda val: f"{val}",
)
def test_remainder(x_shape: tuple, y_shape: tuple):
    def remainder(x: jax.Array, y: jax.Array) -> jax.Array:
        return jlx.rem(x, y)

    run_op_test_with_random_inputs(remainder, [x_shape, y_shape])
