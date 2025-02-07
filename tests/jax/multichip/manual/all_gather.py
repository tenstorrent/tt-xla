# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from infra import run_multichip_test_with_random_inputs, make_partition_spec
import pytest
from utils import compile_fail
from tests.utils import make_partition_spec


@pytest.mark.parametrize(("x_shape", "axis_names"), [((8192, 784), ("batch",))])
@pytest.mark.skip(reason=compile_fail("Multichip still in development"))
def test_all_gather(x_shape: tuple, axis_names: tuple):
    def fwd(batch):
        act = jax.lax.all_gather(batch, axis_names, axis=0, tiled=True)
        return act

    def golden_fwd(batch):
        return jnp.tile(batch, (2, 1))

    in_specs = (make_partition_spec(axis_names),)
    out_specs = make_partition_spec(axis_names)

    run_multichip_test_with_random_inputs(
        fwd, golden_fwd, [x_shape], (2,), axis_names, in_specs, out_specs
    )
