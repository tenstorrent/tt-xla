import os
import jax
import jax.numpy as jnp
import jax._src.xla_bridge as xb
from jax.sharding import Mesh
from jax.sharding import Mesh, PartitionSpec, NamedSharding, SingleDeviceSharding
from jax.sharding import PartitionSpec as P
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

def test_one():
    device_tt = jax.devices('tt')
    print("device:: ", device_tt)
    mesh = jax.make_mesh((1, 2), ('batch', 'model'), devices=device_tt)
    batch = jax.random.normal(jax.random.key(0), (8192, 784))
    W1 = jax.random.normal(jax.random.key(0), (784, 2048))
    B1 = jax.random.normal(jax.random.key(0), (2048))
    out_spec = P('batch')
    @partial(shard_map, mesh=mesh, in_specs=(P('batch', 'model'), P('model', None), P(None)), out_specs=out_spec)
    def fwd(batch, W1_block, B1_block):
        act = jnp.dot(batch, W1_block)
        act = jax.lax.psum(act, 'model')
        act = act + B1_block
        return act
  
    spec = PartitionSpec(None, None)
    spec_2 = PartitionSpec(None)
    output_sharding = NamedSharding(mesh, out_spec)
    batch_sharded = jax.device_put(batch, NamedSharding(mesh, P('batch', 'model')), may_alias=True)
    W1_sharded = jax.device_put(W1, NamedSharding(mesh, P('model', None)), may_alias=True)
    B1_sharded = jax.device_put(B1, NamedSharding(mesh, P(None)), may_alias=True)
    fwd_jit = jax.jit(fwd, out_shardings=output_sharding)
    output = fwd_jit(batch_sharded, W1_sharded, B1_sharded).block_until_ready()
    print(output)

initializePJRT()
n_m = 8192
n_k = 784
n_n = 8192*2

batch_size = 1
layer_sizes = [784, 8192, 8192, 8192, 10]
key = jax.random.key(0)
key, *keys = jax.random.split(key, len(layer_sizes))
k1 = keys[0]
act1 = jax.numpy.ones((256, 512))*2
act2 = jax.numpy.ones((256, 256))
#act = random_input_tensor((n_m, n_k), on_device=True)
#W = random_input_tensor((n_k, n_n), on_device=True)
test_one()