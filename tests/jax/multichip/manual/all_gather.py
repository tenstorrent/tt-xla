# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec
from functools import partial
from infra import run_multichip_test_with_random_inputs
import pytest


@pytest.mark.parametrize(
    ["x_shape", "num_cores"],
    [
        [(8192, 784), 2],
    ],
)
@pytest.mark.skip(reason="Device storage isn't supported")
def test_all_gather_one_axis(x_shape: tuple, num_cores: int):
    def fwd(batch):
        act = jax.lax.all_gather(batch, "batch", axis=0, tiled=True)
        return act

    def golden_fwd(batch):
        return jnp.tile(batch, (num_cores, 1))

    devices = jax.devices("tt")
    mesh = jax.make_mesh((num_cores,), ("batch"), devices=devices)

    in_specs = (PartitionSpec("batch"),)
    out_specs = PartitionSpec("batch")

    run_multichip_test_with_random_inputs(
        fwd, golden_fwd, [x_shape], mesh, in_specs, out_specs
    )

@pytest.mark.parametrize(
    ["x_shape", "mesh_shape"],
    [
        [(8192, 784), (1,2)],
    ],
)
@pytest.mark.skip(reason="Device storage isn't supported issue")
def test_all_gather_two_axis(x_shape: tuple, mesh_shape: tuple):

    def fwd(batch):
        act = jax.lax.all_gather(batch, 'model', axis=1, tiled=True)
        act = jax.lax.all_gather(act, 'batch', axis=0, tiled=True)
        return act
    
    def golden_fwd(batch):
        return jnp.tile(batch, mesh_shape)

    devices = jax.devices("tt")
    mesh = jax.make_mesh(mesh_shape, ('batch', 'model'), devices=devices)

    in_specs = (PartitionSpec('batch', 'model'),)
    out_specs = PartitionSpec('batch', 'model')

    run_multichip_test_with_random_inputs(
        fwd, golden_fwd, [x_shape], mesh, in_specs, out_specs
    )