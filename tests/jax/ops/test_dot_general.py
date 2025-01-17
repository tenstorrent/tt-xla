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
        [(1, 32), (1, 32)],
        [(1, 32, 64), (1, 32, 32)],
        [(2, 32, 64), (2, 32, 64)],
        [(2, 16, 32, 64), (2, 16, 64, 32)],
    ],
)
def test_dot_general(x_shape: tuple, y_shape: tuple):
    def dot_general(x: jax.Array, y: jax.Array) -> jax.Array:
        return jax.lax.dot_general(x, y, dimension_numbers=((1, 1), (0, 0)))

    run_op_test_with_random_inputs(dot_general, [x_shape, y_shape])
