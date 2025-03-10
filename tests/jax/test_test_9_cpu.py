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
flags += " --xla_force_host_platform_device_count=8"  # Simulate 8 devices
# Enforce CPU-only execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_FLAGS"] = flags

def test_one():
    mesh = jax.make_mesh((2, 4), ('batch', 'model'))
    batch = jax.numpy.ones((8192, 784))
    W1 = jax.numpy.ones((784, 2048))
    B1 = jax.numpy.ones((2048))
    out_spec = P('batch')
    @partial(shard_map, mesh=mesh, in_specs=(P('batch', 'model'), P('model', None), P(None)), out_specs=out_spec)
    def fwd(batch, W1, B1):
        act = jnp.dot(batch, W1)
        act = jax.lax.psum(act, 'model')
        act = act + B1
        return act
  
    output_sharding = NamedSharding(mesh, out_spec)
    batch_sharded = jax.device_put(batch, NamedSharding(mesh, P('batch', 'model')), may_alias=True)
    W1_sharded = jax.device_put(W1, NamedSharding(mesh, P('model', None)), may_alias=True)
    B1_sharded = jax.device_put(B1, NamedSharding(mesh, P(None)), may_alias=True)
    fwd_jit = jax.jit(fwd, out_shardings=output_sharding)
    output = fwd_jit(batch_sharded, W1_sharded, B1_sharded).block_until_ready()
    fwd_lowered = fwd_jit.lower(batch_sharded, W1_sharded, B1_sharded)
    fwd_stablehlo = fwd_lowered.compiler_ir(dialect="stablehlo")
    print(fwd_stablehlo)
    print(output)
    print(output.shape)
    
test_one()