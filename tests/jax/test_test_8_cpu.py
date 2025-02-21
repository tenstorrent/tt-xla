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


flags = os.environ.get("XLA_FLAGS", "")
flags += " --xla_force_host_platform_device_count=2"  # Simulate 8 devices
# Enforce CPU-only execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_FLAGS"] = flags

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
    mesh = jax.make_mesh((1, 2), ('batch', 'model'))
    batch = jax.numpy.ones((64, 64))
    W1 = jax.numpy.ones((64, 64))
    print(batch)
    print(W1)
    out_spec = P('batch')


    def fwd(batch, W1_block):
        act = batch + W1_block
        act = jax.numpy.sum(act, axis=0)
        return act
  
    spec_0 = P('batch', 'model')
    spec_1 = P('batch', 'model')
    output_sharding = NamedSharding(mesh, out_spec)
    batch_sharded = jax.device_put(batch, NamedSharding(mesh, spec_0), may_alias=True)
    W1_sharded = jax.device_put(W1, NamedSharding(mesh, spec_1), may_alias=True)
    fwd_jit = jax.jit(fwd, out_shardings=output_sharding)
    output = fwd_jit(batch_sharded, W1_sharded).block_until_ready()
    fwd_lowered = fwd_jit.lower(batch_sharded, W1_sharded)
    fwd_stablehlo = fwd_lowered.compiler_ir(dialect='stablehlo')
    print(fwd_stablehlo)
    print(output)
    print(output.shape)

test_one()