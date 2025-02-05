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
    ["batch_shape", "W1_shape", "B1_shape", "mesh_shape"],
    [
        [(8192, 784), (784, 2048), (2048), (1,2)],
    ],
)
@pytest.mark.skip(reason="Does not compile in TT-mlir")
def test_psum_scatter(batch_shape: tuple, W1_shape: tuple, B1_shape: tuple, mesh_shape: tuple):

    def fwd(batch, W1_block, B1_block):
        act = jnp.dot(batch, W1_block)
        act = jax.lax.psum_scatter(act, 'model', scatter_dimension=1, tiled=True)
        act = act + B1_block
        return act

    def golden_fwd(batch, W1, B1):
        act = jnp.dot(batch, W1)
        act = act + B1
        return act

    devices = jax.devices("tt")
    mesh = jax.make_mesh(mesh_shape, ('batch', 'model'), devices=devices)

    in_specs = (PartitionSpec('batch', 'model'), PartitionSpec('model', None), PartitionSpec('model'),)
    out_specs=PartitionSpec('batch', 'model')

    run_multichip_test_with_random_inputs(
        fwd, golden_fwd, [batch_shape, W1_shape, B1_shape], mesh, in_specs, out_specs, maxval=0.1
    )