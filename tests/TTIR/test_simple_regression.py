# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax

from infrastructure import verify_module


@pytest.mark.parametrize("input_shapes", [[(2, 2)]])
@pytest.mark.skip("Inputs to eltwise binary must be tilized")
def test_gradient(input_shapes):
    def simple_gradient(a):
        def gradient(a):
            return (a**2).sum()

        return jax.grad(gradient)(a)

    verify_module(simple_gradient, input_shapes)


@pytest.mark.parametrize(
    ["weights", "bias", "X", "y"], [[(1, 2), (1, 1), (2, 1), (1, 1)]]
)
@pytest.mark.skip("failed to legalize operation 'stablehlo.dot_general'")
def test_simple_regression(weights, bias, X, y):
    def simple_regression(weights, bias, X, y):
        def loss(weights, bias, X, y):
            predict = X.dot(weights) + bias
            return ((predict - y) ** 2).sum()

        # Compute gradient and update weights
        weights -= jax.grad(loss)(weights, bias, X, y)

        return weights

    verify_module(simple_regression, [weights, bias, X, y])
