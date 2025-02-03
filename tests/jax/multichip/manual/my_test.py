import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec
from functools import partial
from infra import run_multichip_test_with_random_inputs
import pytest

@pytest.mark.parametrize("x_shape", [(256, 256)])
def test_one(x_shape: tuple):

    def fwd(a_block):
        b_block = jnp.negative(a_block)
        stitched_result = jax.lax.psum(b_block, ('x', 'y'))
        return stitched_result
    
    devices = jax.devices('tt')
    mesh = jax.make_mesh((1,2), ('x','y'), devices=devices)
    in_specs = (PartitionSpec('x','y'),)  
    out_specs = PartitionSpec(None, None)

    run_multichip_test_with_random_inputs(fwd, [x_shape], mesh, in_specs, out_specs)

