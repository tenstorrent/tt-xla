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

def initializePJRT():
  path = os.path.join(os.path.dirname(__file__), "/localdev/ajakovljevic/tt-xla/build/src/tt/pjrt_plugin_tt.so")
  if not os.path.exists(path):
    raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}, have you compiled the project?")
  plugin = xb.register_plugin('tt', priority=10, library_path=path, options=None)
  jax.config.update("jax_platforms", "cpu,tt")
  #jax.config.update("jax_use_shardy_partitioner", True)

def test_one():
    mesh = jax.make_mesh((1, 2), ('batch', 'model'))
    batch = jax.numpy.ones((256, 256))
    W1 = jax.numpy.ones((256, 256))
    out_spec = P(None)
    @partial(shard_map, mesh=mesh, in_specs=(P(None, None), P(None, None)), out_specs=out_spec)
    def fwd(batch, W1_block):
        act = jax.numpy.add(batch, W1_block)
        act = jax.lax.psum(act, 'model')
        return act
  
    output_sharding = NamedSharding(mesh, out_spec)
    batch_sharded = jax.device_put(batch, NamedSharding(mesh, P(None, None)), may_alias=True)
    W1_sharded = jax.device_put(W1, NamedSharding(mesh, P(None, None)), may_alias=True)
    fwd_jit = jax.jit(fwd, out_shardings=output_sharding)
    output = fwd_jit(batch_sharded, W1_sharded).block_until_ready()
    print(output)
    print(output.shape)
    

test_one()