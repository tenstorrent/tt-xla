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
        [(32, 64), (64, 32)],
        [(64, 32), (32, 64)],
    ],
)
def test_matmul(x_shape: tuple, y_shape: tuple):
    def matmul(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.matmul(x, y)

    run_op_test_with_random_inputs(matmul, [x_shape, y_shape])