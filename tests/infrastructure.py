import jax
import jax.numpy as jnp

def cpu_random_input(shape, key):
  def random_input(shape, key):
      return jax.random.uniform(jax.random.PRNGKey(key), shape=shape)
  
  cpu_random_input = jax.jit(random_input, static_argnums=[0,1], backend='cpu')
  return cpu_random_input(shape, key)

def verify_module(module, input_shapes, key=42, required_pcc=0.99, required_atol=1e-2):
  tt_device = jax.devices()[0]
  cpu_inputs = [cpu_random_input(shape, key) for shape in input_shapes]
  tt_inputs = [jax.device_put(cpu_input, tt_device) for cpu_input in cpu_inputs]
  tt_graph = jax.jit(module, backend='tt')
  res = tt_graph(*tt_inputs)
  cpu_graph = jax.jit(module, backend='cpu')
  res_cpu = cpu_graph(*cpu_inputs)
  res = jax.device_put(res, res_cpu.device)

  assert res.shape == res_cpu.shape, "Shapes do not match"
  
  if not res.flatten().size == 1: #pcc invalid for scalar values
    pcc = jnp.min(jnp.corrcoef(res.flatten(), res_cpu.flatten()))
    assert pcc >= required_pcc, f"PCC is {pcc} which is less than {required_pcc}"
  
  atol = jnp.max(jnp.abs(res - res_cpu))
  assert atol <= required_atol, f"ATOL is {atol} which is greater than {required_atol}"
  