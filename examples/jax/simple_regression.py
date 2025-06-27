# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#

from jax import grad, jit, vmap
import jax.numpy as jnp
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import jax
import os
import sys
import jax._src.xla_bridge as xb

# Register cpu and tt plugin. tt plugin is registered with higher priority; so
# program will execute on tt device if not specified otherwise.
def initialize():
    backend = "tt"
    path = os.path.join(os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}")

    print("Loading tt_pjrt C API plugin", file=sys.stderr)
    xb.discover_pjrt_plugins()

    plugin = xb.register_plugin("tt", priority=500, library_path=path, options=None)
    print("Loaded", file=sys.stderr)
    jax.config.update("jax_platforms", "tt,cpu")


# Create random inputs (weights) on cpu and move them to tt device if requested.
def random_input_tensor(shape, key=42, on_device=False):
    def random_input(shape, key):
        return jax.random.uniform(jax.random.PRNGKey(key), shape=shape)

    jitted_tensor_creator = jax.jit(random_input, static_argnums=[0, 1], backend="cpu")
    tensor = jitted_tensor_creator(shape, key)
    if on_device:
        tensor = jax.device_put(tensor, jax.devices()[0])
    return tensor


# Predict outcome label.
def predict(params, X):
    w, b = params
    return X.dot(w) + b


# Create a vectorized version of predict function.
batched_predict = vmap(predict, in_axes=(None, 0))


# Calculate loss for give dataset.
def loss(params, X, y):
    pred = batched_predict(params, X)
    return ((pred - y) ** 2).mean()


def test_simple_regression():
    initialize()

    X, y = make_regression(n_samples=150, n_features=2, noise=5)
    y = y.reshape((y.shape[0], 1))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15
    )  # Splitting data into train and test

    Weights = random_input_tensor((X_train.shape[1], 1))
    Bias = 0.0
    l_rate = 0.001
    n_iter = 6000
    size = 127.0
    params = [Weights, Bias]

    gradient = jit(grad(loss), backend="tt")
    print(gradient.lower(params, X_train, y_train).as_text())

    for i in range(n_iter):
        dW, db = gradient(params, X_train, y_train)
        if i % 10 == 0:
            print(f"iteration: {i} {loss(params,X_train,y_train)}")
        weights, bias = params
        weights -= dW * l_rate
        bias -= db * l_rate
        params = [weights, bias]

    test_loss = loss(params, X_test, y_test)  # Model's Loss on test set


if __name__ == "__main__":
    test_simple_regression()
