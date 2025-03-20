# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from infra import make_partition_spec, run_multichip_test_with_random_inputs
import jax
import jax.numpy as jnp
import pytest

from tests.utils import failed_fe_compilation


@pytest.mark.parametrize("use_shardy", [True, False])
@pytest.mark.parametrize(
    ("batch_shape", "W1_shape", "B1_shape", "mesh_shape", "axis_names"),
    [
        ((8192, 784), (784, 2048), (2048), (1, 2), ("batch", "model")),
    ],
)
@pytest.mark.skip(reason=failed_fe_compilation("Multichip still in development"))
def test_psum_scatter(
    batch_shape: tuple,
    W1_shape: tuple,
    B1_shape: tuple,
    mesh_shape: tuple,
    axis_names: tuple,
    use_shardy: bool,
):
    def fwd(batch, W1_block, B1_block):
        act = jnp.dot(batch, W1_block)
        act = jax.lax.psum_scatter(act, "model", scatter_dimension=1, tiled=True)
        act = act + B1_block
        return act

    in_specs = (
        make_partition_spec(axis_names),
        make_partition_spec((axis_names[1], None)),
        make_partition_spec((axis_names[1],)),
    )
    out_specs = make_partition_spec(axis_names)

    run_multichip_test_with_random_inputs(
        fwd,
        [batch_shape, W1_shape, B1_shape],
        mesh_shape,
        axis_names,
        in_specs,
        out_specs,
        use_shardy=use_shardy,
        maxval=0.1,
    )
