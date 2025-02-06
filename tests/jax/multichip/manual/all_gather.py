# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from infra import run_multichip_test_with_random_inputs
import pytest
from tests.utils import compile_failed, make_partition_spec


@pytest.mark.parametrize("x_shape", [(8192, 784)])
@pytest.mark.skip(reason=compile_failed("Multichip still in development"))
def test_all_gather(x_shape: tuple):
    def fwd(batch):
        act = jax.lax.all_gather(batch, "batch", axis=0, tiled=True)
        return act

    def golden_fwd(batch):
        return jnp.tile(batch, (2, 1))

    in_specs = (make_partition_spec(("batch")),)
    out_specs = make_partition_spec(("batch"))

    run_multichip_test_with_random_inputs(
        fwd, golden_fwd, [x_shape], (2,), ("batch"), in_specs, out_specs
    )
