import os
import jax
import jax.numpy as jnp
import jax._src.xla_bridge as xb
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from functools import partial
def random_input_tensor(shape, key=42, on_device=False, dtype=jnp.float32):
  device_cpu = jax.devices('cpu')[0]
  with jax.default_device(device_cpu):
    tensor = jax.random.uniform(jax.random.PRNGKey(key), shape=shape, dtype=dtype)
  # Although the random tensor is generated on cpu but it is not committed to
  # cpu; so this tensor can be moved to the device and subsequent code will
  # execute on the device. Placing the generated tensor explicitly to cpu or
  # device to avoid unwanted behavior.
  if on_device:
    tensor = jax.device_put(tensor, jax.devices('tt')[0])
  else:
    tensor = jax.device_put(tensor, device_cpu)
  return tensor

def initializePJRT():
  path = os.path.join(os.path.dirname(__file__), "/localdev/ajakovljevic/tt-xla/build/src/tt/pjrt_plugin_tt.so")
  if not os.path.exists(path):
    raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}, have you compiled the project?")
  plugin = xb.register_plugin('tt', priority=10, library_path=path, options=None)
  jax.config.update("jax_platforms", "cpu,tt")
  #jax.config.update("jax_use_shardy_partitioner", True)

def mesh_matmul(act, W):
  device_tt = jax.devices('tt')
  print("device:: ", device_tt)
  devices = mesh_utils.create_device_mesh((1, 1), device_tt)
  #devices = mesh_utils.create_device_mesh((1, 2), device_tt)
  #devices = mesh_utils.create_device_mesh((2, 4), device_tt)
  #devices = mesh_utils.create_device_mesh((1, 8), device_tt)
  #devices = mesh_utils.create_device_mesh((4, 2), device_tt)
  mesh = Mesh(devices=devices, axis_names=('x', 'y'))
  P = PartitionSpec
  @partial(shard_map, mesh=mesh, in_specs=(P(None, 'x', 'y'), P(None, 'y', None)),
           out_specs=P(None, None, 'x', None))
  def matmul_basic(a_block, b_block):
    c_partialsum = jnp.dot(a_block, b_block)
    c_block = jax.lax.psum(c_partialsum, 'y')
    return c_block
  #result = matmul_basic(act, W)
  #print(result)
  lowered_single = jax.jit(matmul_basic).lower(act, W)
  print(lowered_single.as_text())
initializePJRT()
n_m = 8192
n_k = 784
n_n = 8192*2
batch_size = 1
layer_sizes = [784, 8192, 8192, 8192, 10]
key = jax.random.key(0)
key, *keys = jax.random.split(key, len(layer_sizes))
k1 = keys[0]
act = jax.random.normal(k1, (1, n_m, n_k))
W = jax.random.normal(k1, (1, n_k, n_n))
#act = random_input_tensor((n_m, n_k), on_device=True)
#W = random_input_tensor((n_k, n_n), on_device=True)
mesh_matmul(act, W)