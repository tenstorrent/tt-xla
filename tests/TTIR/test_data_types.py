# RUN: %PYTHON %s | FileCheck %s


import jax
import jax.numpy as jnp

# Currently, tt::runtime only support float32, bfloat16, uint16, and uint32

def test_data_types(capfd):
  print("Starting test")
  a = jnp.array([[1., 2.], [3., 4.]], dtype=jnp.float32)
  b = jnp.array([[5., 6.], [7., 8.]], dtype=jnp.bfloat16)
  c = jnp.array([[1, 2], [3, 4]], dtype=jnp.uint32)
  d = jnp.array([[5, 6], [7, 8]], dtype=jnp.uint16)
  print(a)
  out, _ = capfd.readouterr()
  assert "[[1. 2.]\n [3. 4.]]" in out
  # CHECK: [[1. 2.]
  # CHECK:  [3. 4.]]


  print(b)
  out, _ = capfd.readouterr()
  assert "[[5 6]\n [7 8]]" in out
  # CHECK: [[5 6]
  # CHECK:  [7 8]]

  print(c)
  out, _ = capfd.readouterr()
  assert "[[1 2]\n [3 4]]" in out
  # CHECK: [[1 2]
  # CHECK:  [3 4]]

  print(d)
  out, _ = capfd.readouterr()
  assert "[[5 6]\n [7 8]]" in out
  # CHECK: [[5 6]
  # CHECK:  [7 8]]
