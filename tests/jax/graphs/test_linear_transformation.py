# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import run_graph_test_with_random_inputs


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize(
    ["x_shape", "y_shape", "bias_shape"],
    [
        [(32, 32), (32, 32), (1, 32)],
        [(64, 64), (64, 64), (1, 64)],
        [(32, 64), (64, 32), (1, 32)],
        [(64, 32), (32, 64), (1, 64)],
    ],
)
def test_linear_transformation(
    x_shape: tuple, y_shape: tuple, bias_shape: tuple
):
    def linear_transformation(
        x: jax.Array, y: jax.Array, bias: jax.Array
    ) -> jax.Array:
        return jnp.matmul(x, y) + bias

    run_graph_test_with_random_inputs(
        linear_transformation, [x_shape, y_shape, bias_shape]
    )
