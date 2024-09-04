# RUN: %PYTHON %s | FileCheck %s


import jax
import jax.numpy as jnp


def test_2x2_array_add():
  def module_add(a, b):
    return a + b
  print("Starting test")
  a = jnp.array([[1., 2.], [3., 4.]])
  b = jnp.array([[5., 6.], [7., 8.]])
  graph = jax.jit(module_add)
  res = graph(a, b)
  print(res)

  # CHECK: [ 6. 8.]
  # CHECK: [10. 12.]

def test_3x2_array_add():
  def module_add(a, b):
    return a + b
  print("Starting test")
  a = jnp.array([[1., 2., 6.], [3., 4., 8.]])
  b = jnp.array([[5., 6., 2.], [7., 8., 4.]])
  graph = jax.jit(module_add)
  res = graph(a, b)
  print(res)

  # CHECK: [ 6. 8. 8.]
  # CHECK: [10. 12. 12.]


def test_module_add():
  def module_add(a, b):
    c = a + a
    d = b + b
    return c + d

  print("Starting test")
  a = jnp.array([[1., 2.], [3., 4.]])
  b = jnp.array([[5., 6.], [7., 8.]])
  graph = jax.jit(module_add)
  res = graph(a, b)
  print(res)
  # CHECK: [12. 16.]
  # CHECK: [20. 24.]
