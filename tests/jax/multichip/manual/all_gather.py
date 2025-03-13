# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from infra import run_multichip_test_with_random_inputs, make_partition_spec
import pytest
from utils import compile_fail


@pytest.mark.parametrize(
    ("x_shape", "mesh_shape", "axis_names"), [((8192, 784), (2,), ("batch",))]
)
@pytest.mark.skip(reason=compile_fail("Multichip still in development"))
def test_all_gather(x_shape: tuple, mesh_shape: tuple, axis_names: tuple):
    def fwd(batch):
        act = jax.lax.all_gather(batch, axis_names, axis=0, tiled=True)
        return act

    in_specs = (make_partition_spec(axis_names),)
    out_specs = make_partition_spec(axis_names)

    run_multichip_test_with_random_inputs(
        fwd, [x_shape], mesh_shape, axis_names, in_specs, out_specs
    )
