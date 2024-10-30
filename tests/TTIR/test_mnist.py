# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax
import jax.numpy as jnp

from infrastructure import verify_test_module


@pytest.mark.skip("VHLO Legalization failed.")
@verify_test_module([(32, 32), (32, 32)], required_atol=3e-2)
def test_matmul(a, b):
    return jnp.matmul(a, b)


@pytest.mark.skip("VHLO Legalization failed.")
@verify_test_module([(32, 32), (32, 32), (1, 32)], required_atol=3e-2)
def test_matmul_with_bias(a, b, bias):
    return jnp.matmul(a, b) + bias


@verify_test_module([(32, 32), (32, 32)])
def test_relu_no_broadcast(a, b):
    return jnp.maximum(a, b)


@pytest.mark.skip("Asserts")
@verify_test_module([(32, 32)])
def test_relu(a):
    return jnp.maximum(a, 0)


@pytest.mark.skip("keepdims=False in runtime")
@verify_test_module([(32, 32)])
def test_softmax(a):
    return jax.nn.softmax(a)


@pytest.mark.skip(
    "Index is out of bounds for the rank, should be between 0 and 0 however is 18446744073709551615"
)
@verify_test_module([(32, 784), (784, 128), (1, 128), (128, 128), (1, 128), (128, 10), (1, 10)])
def test_mnist(act, w0, b0, w1, b1, w2, b2):
    x = jnp.matmul(act, w0) + b0
    x = jnp.maximum(x, 0)
    x = jnp.matmul(x, w1) + b1
    x = jnp.maximum(x, 0)
    x = jnp.matmul(x, w2) + b2
    x = jax.nn.softmax(x)
    return x
