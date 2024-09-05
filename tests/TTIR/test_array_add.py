import jax
import jax.numpy as jnp

from infrastructure import verify_module

def test_2x2_array_add():
  def module_add(a, b):
    return a + b

  verify_module(module_add, [(2, 2), (2, 2)])


def test_3x2_array_add():
  def module_add(a, b):
    return a + b

  verify_module(module_add, [(3, 2), (3, 2)])


def test_module_add():
  def module_add(a, b):
    c = a + a
    d = b + b
    return c + d

  verify_module(module_add, [(32, 32), (32, 32)])
