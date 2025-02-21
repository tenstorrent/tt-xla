import os

# Set this to True to run the model on CPU only.
USE_CPU_ONLY = True

flags = os.environ.get("XLA_FLAGS", "")
if USE_CPU_ONLY:
    flags += " --xla_force_host_platform_device_count=8"  # Simulate 8 devices
    # Enforce CPU-only execution
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    # GPU flags
    flags += (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=false "
        "--xla_gpu_enable_async_collectives=true "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_highest_priority_async_stream=true "
    )
os.environ["XLA_FLAGS"] = flags

import functools
from typing import Any, Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from functools import partial

PyTree = Any
Metrics = Dict[str, Tuple[jax.Array, ...]]

mesh = jax.make_mesh((4, 2), ('batch', 'model'))
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

output_sharding = NamedSharding(mesh, out_spec)
fwd_jit = jax.jit(fwd, out_shardings=output_sharding)
fwd_lowered = fwd_jit.lower(batch, W1, B1)
fwd_stablehlo = fwd_lowered.compiler_ir(dialect='stablehlo')
print(fwd_stablehlo.dump())
output = fwd_jit(batch, W1, B1).block_until_ready()

jax.debug.visualize_array_sharding(output)

golden = jnp.dot(batch, W1) + B1
print(output)
print(jnp.allclose(output, golden, atol=1e-4))
jax.debug.visualize_array_sharding(golden)