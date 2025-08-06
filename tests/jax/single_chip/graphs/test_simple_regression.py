# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from infra import run_graph_test_with_random_inputs
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize(
    ["weights", "bias", "X", "y"], [[(1, 2), (1, 1), (2, 1), (1, 1)]]
)
def test_simple_regression(weights, bias, X, y):
    def simple_regression(weights, bias, X, y):
        def loss(weights, bias, X, y):
            predict = X.dot(weights) + bias if bias is not None else X.dot(weights)
            return ((predict - y) ** 2).sum()

        # Compute gradient and update weights.
        weights -= jax.grad(loss)(weights, bias, X, y)
        return weights

    run_graph_test_with_random_inputs(simple_regression, [weights, bias, X, y])
