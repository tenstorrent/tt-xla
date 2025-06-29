# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.divide",
    shlo_op_name="stablehlo.divide",
)
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
    ids=lambda val: f"{val}",
)
def test_divide(x_shape: tuple, y_shape: tuple):
    def divide(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.divide(x, y)

    run_op_test_with_random_inputs(divide, [x_shape, y_shape])
