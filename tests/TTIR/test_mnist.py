# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax
import jax.numpy as jnp

from infrastructure import verify_module

def test_matmul():
  def module_matmul(a, b):
    return jnp.matmul(a, b)

  verify_module(module_matmul, [(32, 32), (32, 32)], required_atol=3e-2)

def test_matmul_with_bias():
  def module_matmul(a, b, bias):
    return jnp.matmul(a, b) + bias

  verify_module(module_matmul, [(32, 32), (32, 32), (1, 32)], required_atol=3e-2)

def test_relu_no_broadcast():
  def module_relu(a, b):
    return jnp.maximum(a, b)

  verify_module(module_relu, [(32, 32), (32, 32)])


def test_relu():
  def module_relu(a):
    return jnp.maximum(0, a)

  verify_module(module_relu, [(32, 32)])

def test_softmax():
  def module_softmax(a):
    return jax.nn.softmax(a)

  verify_module(module_softmax, [(32, 32)])

def test_mnist():
  def module_mnist(act, w0, b0, w1, b1, w2, b2):
    x = jnp.matmul(act, w0) + b0
    x = jnp.maximum(0, x)
    x = jnp.matmul(x, w1) + b1
    x = jnp.maximum(0, x)
    x = jnp.matmul(x, w2) + b2
    x = jax.nn.softmax(x)
    return x

  verify_module(module_mnist, [(32, 784), (784, 128), (1, 128), (128, 128), (1, 128), (128, 10), (1, 10)])
  