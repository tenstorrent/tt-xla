import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec
from functools import partial
from infra import run_multichip_test_with_random_inputs
import pytest

@pytest.mark.parametrize("x_shape", [(8192, 784)])
def test_one(x_shape: tuple):
    def fwd(batch):
        act = jax.lax.all_gather(batch, 'batch', axis=0, tiled=True)
        return act

    def golden_fwd(batch):
        return jnp.tile(batch, (2, 1))
    
    devices = jax.devices('tt')
    mesh = jax.make_mesh((2,), ('batch'), devices=devices)

    in_specs = (PartitionSpec('batch'),)
    out_specs = PartitionSpec('batch')

    run_multichip_test_with_random_inputs(fwd, [x_shape], mesh, in_specs, out_specs)

