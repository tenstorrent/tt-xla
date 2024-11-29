# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax
import jax.numpy as jnp

from infrastructure import verify_module


@pytest.mark.parametrize("input_shapes", [[(32, 32), (32, 32)]])
def test_matmul(input_shapes):
    def module_matmul(a, b):
        return jnp.matmul(a, b)

    verify_module(module_matmul, input_shapes, required_atol=3e-2)


@pytest.mark.parametrize("input_shapes", [[(32, 32), (32, 32), (1, 32)]])
def test_matmul_with_bias(input_shapes):
    def module_matmul(a, b, bias):
        return jnp.matmul(a, b) + bias

    verify_module(module_matmul, input_shapes, required_atol=3e-2)


@pytest.mark.parametrize("input_shapes", [[(32, 32), (32, 32)]])
def test_relu_no_broadcast(input_shapes):
    def module_relu(a, b):
        return jnp.maximum(a, b)

    verify_module(module_relu, input_shapes)


@pytest.mark.parametrize("input_shapes", [[(32, 32)]])
@pytest.mark.skip(
    "ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast"
)
def test_relu(input_shapes):
    def module_relu(a):
        return jnp.maximum(a, 0)

    verify_module(module_relu, input_shapes)


@pytest.mark.parametrize("input_shapes", [[(32, 32)]])
@pytest.mark.skip("keepdim=False is not supported")
def test_softmax(input_shapes):
    def module_softmax(a):
        return jax.nn.softmax(a)

    verify_module(module_softmax, input_shapes)


@pytest.mark.parametrize(
    ["act", "w0", "b0", "w1", "b1", "w2", "b2"],
    [[(32, 784), (784, 128), (1, 128), (128, 128), (1, 128), (128, 10), (1, 10)]],
)
@pytest.mark.skip(
    "ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast"
)
def test_mnist(act, w0, b0, w1, b1, w2, b2):
    def module_mnist(act, w0, b0, w1, b1, w2, b2):
        x = jnp.matmul(act, w0) + b0
        x = jnp.maximum(x, 0)
        x = jnp.matmul(x, w1) + b1
        x = jnp.maximum(x, 0)
        x = jnp.matmul(x, w2) + b2
        x = jax.nn.softmax(x)
        return x

    verify_module(
        module_mnist,
        [act, w0, b0, w1, b1, w2, b2],
    )
