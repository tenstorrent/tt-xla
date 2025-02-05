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


@pytest.mark.parametrize("x_shape", [(256, 256)])
@pytest.mark.skip(reason="Multichip still in development")
def test_unary_eltwise(x_shape: tuple):
    def fwd(a_block):
        b_block = jnp.negative(a_block)
        stitched_result = jax.lax.psum(b_block, ("x", "y"))
        return stitched_result

    def fwd_single_device(a_block):
        a1, a2 = jnp.split(a_block, 2, axis=1)

        b1, b2 = jnp.negative(a1), jnp.negative(a2)

        stitched_result = b1 + b2
        return stitched_result

    devices = jax.devices("tt")
    mesh = jax.make_mesh((1, 2), ("x", "y"), devices=devices)
    in_specs = (PartitionSpec("x", "y"),)
    out_specs = PartitionSpec(None, None)

    run_multichip_test_with_random_inputs(
        fwd, fwd_single_device, [x_shape], mesh, in_specs, out_specs
    )
