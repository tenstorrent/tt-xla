# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import pytest
from infra import ComparisonConfig, run_graph_test_with_random_inputs


@pytest.fixture
def comparison_config() -> ComparisonConfig:
    config = ComparisonConfig()
    config.pcc.allclose.atol = 0.03
    config.pcc.allclose.rtol = 0.03
    return config


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize(
    ["W1", "b1", "W2", "b2", "X", "y"],
    [
        [(784, 64), (32, 64), (64, 10), (32, 10), (32, 784), (32, 10)]
    ],  # 32 samples, 784 features (28x28), 10 output classes
)
def test_nn_with_relu(
    W1, b1, W2, b2, X, y, comparison_config: ComparisonConfig
):
    def simple_nn(W1, b1, W2, b2, X, y):
        def forward(W1, b1, W2, b2, X):
            hidden = jnp.dot(X, W1) + b1
            hidden = jnp.maximum(0, hidden)
            output = jnp.dot(hidden, W2) + b2
            return output

        def loss(W1, b1, W2, b2, X, y):
            output = forward(W1, b1, W2, b2, X)
            return jnp.mean((output - y) ** 2)

        @jax.jit
        def update_params(W1, b1, W2, b2, X, y, lr=0.01):
            grads = jax.grad(loss, argnums=(0, 1, 2, 3))(W1, b1, W2, b2, X, y)
            W1 -= lr * grads[0]
            b1 -= lr * grads[1]
            W2 -= lr * grads[2]
            b2 -= lr * grads[3]
            return W1, b1, W2, b2, grads

        for i in range(50):
            W1, b1, W2, b2, grads = update_params(W1, b1, W2, b2, X, y, lr=0.01)

        final_loss = loss(W1, b1, W2, b2, X, y)
        return final_loss

    run_graph_test_with_random_inputs(
        simple_nn, [W1, b1, W2, b2, X, y], comparison_config=comparison_config
    )
