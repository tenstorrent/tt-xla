# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from infra import run_multichip_test_with_random_inputs, make_partition_spec
from infra.utils import add_use_shardy_flag
import jax
import jax.numpy as jnp
import pytest
from tests.utils import failed_fe_compilation


@pytest.mark.parametrize(
    ("x_shape", "mesh_shape", "axis_names", "use_shardy"),
    add_use_shardy_flag([((256, 256), (1, 2), ("x", "y"))]),
)
@pytest.mark.skip(reason=failed_fe_compilation("Multichip still in development"))
def test_unary_eltwise(
    x_shape: tuple, mesh_shape: tuple, axis_names: tuple, use_shardy: bool
):
    def fwd(a_block):
        b_block = jnp.negative(a_block)
        return b_block

    def fwd_single_device(a_block):
        a1, a2 = jnp.split(a_block, 2, axis=1)

        b1, b2 = jnp.negative(a1), jnp.negative(a2)
        stitched_result = jnp.concatenate([b1, b2], axis=1)

        return stitched_result

    in_specs = (make_partition_spec(axis_names),)
    out_specs = make_partition_spec((axis_names))

    run_multichip_test_with_random_inputs(
        fwd,
        fwd_single_device,
        [x_shape],
        mesh_shape,
        axis_names,
        in_specs,
        out_specs,
        use_shardy,
    )
