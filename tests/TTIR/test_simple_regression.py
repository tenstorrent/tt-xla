# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax

from infrastructure import verify_test_module


@pytest.mark.skip(
    "Module contains function used inside the main function. Cannot compile Flatbuffer."
)
def test_gradient():
    @verify_test_module([(2, 2)])
    def simple_gradient(a):
        def gradient(a):
            return (a**2).sum()

        return jax.grad(gradient)(a)

    simple_gradient()


@pytest.mark.skip("TT_METAL_HOME is not set.")
@verify_test_module([(1, 2), (1, 1), (2, 1), (1, 1)])
def test_simple_regression():
    def simple_regression(weights, bias, X, y):
        def loss(weights, bias, X, y):
            predict = X.dot(weights) + bias
            return ((predict - y) ** 2).sum()

        # Compute gradient and update weights
        weights -= jax.grad(loss)(weights, bias, X, y)

        return weights

    simple_regression()
