# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax
import jax.numpy as jnp

from infrastructure import verify_module


@pytest.mark.xfail
def test_gradient():
  def simple_gradient(a):
    def gradient(a):
      return (a ** 2).sum()

    return jax.grad(gradient)(a)

  verify_module(simple_gradient, [(2, 2)])


@pytest.mark.xfail
def test_simple_regression():
  def simple_regression(weights, bias, X, y):
    def loss(weights, bias, X, y):
      predict = X.dot(weights) + bias
      return ((predict - y) ** 2).sum()

    # Compute gradient and update weights
    weights -= jax.grad(loss)(weights, bias, X, y)

    return weights
    
  verify_module(simple_regression, [(1, 2), (1,1), (2, 1), (1, 1)])

