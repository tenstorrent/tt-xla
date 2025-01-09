# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs


@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
)
def test_multiply(x_shape: tuple, y_shape: tuple):
    def multiply(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.multiply(x, y)

    run_op_test_with_random_inputs(multiply, [x_shape, y_shape])
