# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#

import os
import sys

import jax
import jax._src.xla_bridge as xb
import jax.numpy as jnp
from jax import grad, jit, vmap
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


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
    X, y = make_regression(n_samples=150, n_features=2, noise=5)
    y = y.reshape((y.shape[0], 1))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15
    )  # Splitting data into train and test

    Weights = random_input_tensor((X_train.shape[1], 1), on_device=True)
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
