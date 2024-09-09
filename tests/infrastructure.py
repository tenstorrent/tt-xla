import jax
import jax.numpy as jnp

def random_input_tensor(shape, key=42, on_device=False):
  def random_input(shape, key):
      return jax.random.uniform(jax.random.PRNGKey(key), shape=shape)
  
  jitted_tensor_creator = jax.jit(random_input, static_argnums=[0,1], backend='cpu')
  tensor = jitted_tensor_creator(shape, key)
  if on_device:
    tensor = jax.device_put(tensor, jax.devices()[0])
  return tensor

def compare_tensor_to_golden(tensor, golden, required_pcc=0.99, required_atol=1e-2, assert_on_error=True):
  ret = True
  if tensor.device != golden.device:
    tensor = jax.device_put(tensor, golden.device)
  
  ret = ret and tensor.shape == golden.shape
  if assert_on_error:
    assert ret, "Shapes do not match"
  
  if not tensor.flatten().size == 1: #pcc invalid for scalar values
    pcc = jnp.min(jnp.corrcoef(tensor.flatten(), golden.flatten()))
    ret = ret and pcc >= required_pcc
    if assert_on_error:
      assert ret, f"PCC is {pcc} which is less than {required_pcc}"
  
  atol = jnp.max(jnp.abs(tensor - golden))
  ret = ret and atol <= required_atol
  if assert_on_error:
    assert ret, f"ATOL is {atol} which is greater than {required_atol}"
  
  return ret

def verify_module(module, input_shapes, key=42, required_pcc=0.99, required_atol=1e-2):
  tt_device = jax.devices()[0]
  cpu_inputs = [random_input_tensor(shape, key) for shape in input_shapes]
  tt_inputs = [jax.device_put(cpu_input, tt_device) for cpu_input in cpu_inputs]
  graph = jax.jit(module)
  res = graph(*tt_inputs)
  res_cpu = graph(*cpu_inputs)
  
  compare_tensor_to_golden(res, res_cpu, required_pcc, required_atol)
