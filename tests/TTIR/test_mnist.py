import pytest
import jax
import jax.numpy as jnp

from infrastructure import verify_module

@pytest.mark.xfail
def test_matmul():
  def module_matmul(a, b):
    return jnp.matmul(a, b)

  verify_module(module_matmul, [(32, 32), (32, 32)])

@pytest.mark.xfail
def test_matmul_with_bias():
  def module_matmul(a, b, bias):
    return jnp.matmul(a, b) + bias

  verify_module(module_matmul, [(32, 32), (32, 32), (1, 32)])

@pytest.mark.xfail
def test_relu():
  def module_relu(a):
    return jnp.maximum(a, 0)

  verify_module(module_relu, [(32, 32)])

@pytest.mark.xfail
def test_softmax():
  def module_softmax(a):
    return jax.nn.softmax(a)

  verify_module(module_softmax, [(32, 32)])

@pytest.mark.xfail
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