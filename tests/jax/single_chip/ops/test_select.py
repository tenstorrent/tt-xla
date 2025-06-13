# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs

from tests.utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.where",
    shlo_op_name="stablehlo.select",
)
@pytest.mark.parametrize(
    ["cond_shape", "x_shape", "y_shape"],
    [
        [(32, 32), (32, 32), (32, 32)],
        [(64, 64), (64, 64), (64, 64)],
        [(16, 32), (16, 32), (16, 32)],
    ],
    ids=lambda val: f"{val}",
)
def test_select(cond_shape: tuple, x_shape: tuple, y_shape: tuple):
    def select(cond: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.where(cond, x, y)

    run_op_test_with_random_inputs(select, [cond_shape, x_shape, y_shape])
