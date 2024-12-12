# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from infra.comparison import ComparisonConfig
from infra.tester import run_graph_test_with_random_inputs
from jax import numpy as jnp


@pytest.fixture
def comparison_config() -> ComparisonConfig:
    config = ComparisonConfig()
    config.disable_all()
    config.allclose.enable()
    return config


def arbitrary_op_chain(x: jax.Array, y: jax.Array) -> jax.Array:
    a = jnp.abs(x)
    b = jnp.add(a, y)
    c = jnp.divide(a, b)
    return jnp.exp(c)


@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(3, 3), (3, 3)],
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
)
def test_arbitrary_op_chain(
    x_shape: tuple, y_shape: tuple, comparison_config: ComparisonConfig
):
    run_graph_test_with_random_inputs(
        arbitrary_op_chain, [x_shape, y_shape], comparison_config
    )
