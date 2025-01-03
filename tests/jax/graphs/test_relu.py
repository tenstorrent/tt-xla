# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import random_tensor, run_graph_test, run_graph_test_with_random_inputs


@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
)
def test_maximum_as_relu(x_shape: tuple, y_shape: tuple):
    """
    Special case where maximum op is used as RELU activation function where second
    argument is a **tensor** of zeros.
    """

    def maximum(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.maximum(x, y)

    x, y = random_tensor(x_shape), jnp.zeros(y_shape)
    run_graph_test(maximum, [x, y])


@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)])
@pytest.mark.skip(
    "ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast"
)
def test_relu(x_shape: tuple):
    """
    Special case of RELU activation function where second argument is a 0 (scalar).
    """

    def relu(x: jax.Array) -> jax.Array:
        return jnp.maximum(x, 0)

    run_graph_test_with_random_inputs(relu, [x_shape])
