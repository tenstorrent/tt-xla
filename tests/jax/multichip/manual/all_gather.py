# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import make_partition_spec, run_multichip_test_with_random_inputs

from tests.utils import failed_fe_compilation


@pytest.mark.parametrize(
    ("x_shape", "mesh_shape", "axis_names"), [((8192, 784), (2,), ("batch",))]
)
@pytest.mark.skip(reason=failed_fe_compilation("Multichip still in development"))
def test_all_gather(x_shape: tuple, mesh_shape: tuple, axis_names: tuple):
    def fwd(batch):
        act = jax.lax.all_gather(batch, axis_names, axis=0, tiled=True)
        return act

    def golden_fwd(batch):
        return jnp.tile(batch, (2, 1))

    in_specs = (make_partition_spec(axis_names),)
    out_specs = make_partition_spec(axis_names)

    run_multichip_test_with_random_inputs(
        fwd, golden_fwd, [x_shape], mesh_shape, axis_names, in_specs, out_specs
    )
