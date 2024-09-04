import jax
import jax.numpy as jnp

def test_2x2_array_add():
  def module_add(a, b):
    return a + b

  a = jnp.array([[1., 2.], [3., 4.]])
  b = jnp.array([[5., 6.], [7., 8.]])
  tt_graph = jax.jit(module_add)
  res = tt_graph(a, b)
  cpu_graph = jax.jit(module_add, backend='cpu')
  res_cpu = cpu_graph(a, b)
  assert jnp.allclose(res, res_cpu)


def test_3x2_array_add():
  def module_add(a, b):
    return a + b

  a = jnp.array([[1., 2., 6.], [3., 4., 8.]])
  b = jnp.array([[5., 6., 2.], [7., 8., 4.]])
  tt_graph = jax.jit(module_add)
  res = tt_graph(a, b)
  cpu_graph = jax.jit(module_add, backend='cpu')
  res_cpu = cpu_graph(a, b)
  assert jnp.allclose(res, res_cpu)


def test_module_add():
  def module_add(a, b):
    c = a + a
    d = b + b
    return c + d

  a = jnp.array([[1., 2.], [3., 4.]])
  b = jnp.array([[5., 6.], [7., 8.]])
  tt_graph = jax.jit(module_add)
  res = tt_graph(a, b)
  cpu_graph = jax.jit(module_add, backend='cpu')
  res_cpu = cpu_graph(a, b)
  assert jnp.allclose(res, res_cpu)
