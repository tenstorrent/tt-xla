import pytest
import jax
import jax.numpy as jnp
from infrastructure import verify_module

def test_1d_array_add():
  def module_add(a, b):
    return a + b

  verify_module(module_add, [(1,), (1,)])
