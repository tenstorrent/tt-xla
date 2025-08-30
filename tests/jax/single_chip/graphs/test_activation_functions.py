# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import run_graph_test_with_random_inputs
from utils import Category


# @pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)])
def test_relu(x_shape: tuple):
    """Test ReLU activation function."""

    def relu(x: jax.Array) -> jax.Array:
        return jnp.maximum(x, 0)

    run_graph_test_with_random_inputs(relu, [x_shape])


# TODO add tests for other activations functions.
