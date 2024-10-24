# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax
import jax.numpy as jnp

from infrastructure import verify_module

@pytest.mark.skip("VHLO Legalization failed.")
def test_matmul():
  def module_matmul(a, b):
    return jnp.matmul(a, b)

  verify_module(module_matmul, [(32, 32), (32, 32)], required_atol=3e-2)

@pytest.mark.skip("VHLO Legalization failed.")
def test_matmul_with_bias():
  def module_matmul(a, b, bias):
    return jnp.matmul(a, b) + bias

  verify_module(module_matmul, [(32, 32), (32, 32), (1, 32)], required_atol=3e-2)

def test_relu_no_broadcast():
  def module_relu(a, b):
    return jnp.maximum(a, b)

  verify_module(module_relu, [(32, 32), (32, 32)])


def test_relu():
  pytest.skip("Asserts")
  def module_relu(a):
    return jnp.maximum(a, 0)

  verify_module(module_relu, [(32, 32)])

@pytest.mark.skip("keepdims=False in runtime")
def test_softmax():
  def module_softmax(a):
    return jax.nn.softmax(a)

  verify_module(module_softmax, [(32, 32)])

@pytest.mark.skip("Index is out of bounds for the rank, should be between 0 and 0 however is 18446744073709551615")
def test_mnist():
  def module_mnist(act, w0, b0, w1, b1, w2, b2):
    x = jnp.matmul(act, w0) + b0
    x = jnp.maximum(x, 0)
    x = jnp.matmul(x, w1) + b1
    x = jnp.maximum(x, 0)
    x = jnp.matmul(x, w2) + b2
    x = jax.nn.softmax(x)
    return x

  verify_module(module_mnist, [(32, 784), (784, 128), (1, 128), (128, 128), (1, 128), (128, 10), (1, 10)])
  
