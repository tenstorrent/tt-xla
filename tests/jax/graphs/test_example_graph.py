# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from infra import run_graph_test_with_random_inputs
from jax import numpy as jnp


def example_graph(x: jax.Array, y: jax.Array) -> jax.Array:
    a = jnp.abs(x)
    b = jnp.add(a, y)
    c = jnp.divide(a, b)
    return jnp.exp(c)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(3, 3), (3, 3)],
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
)
def test_example_graph(x_shape: tuple, y_shape: tuple):
    run_graph_test_with_random_inputs(example_graph, [x_shape, y_shape])
