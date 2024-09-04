# RUN: %PYTHON %s | FileCheck %s

import pytest
import jax
import jax.numpy as jnp

def test_scalar_add():
  pytest.skip("Not working")
  def module_add(a, b):
    return a + b

  a = jnp.float32(5.)
  b = jnp.array(6.)
  graph = jax.jit(module_add)
  res = graph(a, b)
  print(res)

  # CHECK: 11.
