# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from infra import make_partition_spec, run_multichip_test_with_random_inputs
import jax
import jax.numpy as jnp
import pytest
from infra import (
    make_partition_spec,
    MultichipMode,
    run_multichip_test_with_random_inputs,
)

from tests.utils import failed_ttmlir_compilation


@pytest.mark.n300
@pytest.mark.push
@pytest.mark.parametrize(
    "use_shardy",
    [
        pytest.param(
            True,
            marks=pytest.mark.skip(reason="Shardy sharding not supported (issue #383)"),
        ),
        False,
    ],
)
@pytest.mark.parametrize(
    ("batch_shape", "W1_shape", "B1_shape", "mesh_shape", "axis_names"),
    [
        ((8192, 784), (784, 2048), (2048), (1, 2), ("batch", "model")),
    ],
)
@pytest.mark.parametrize(
    "multichip_mode",
    [
        MultichipMode.FULLY_MANUAL,
        MultichipMode.MANUAL,
        MultichipMode.AUTOMATIC,
    ],
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "Coordinate MeshCoordinate([1, 0]) is out of bounds for shape MeshShape([1, 2]) "
        "(https://github.com/tenstorrent/tt-xla/issues/381)"
    )
)
def test_psum_scatter(
    use_shardy: bool,
    batch_shape: tuple,
    W1_shape: tuple,
    B1_shape: tuple,
    mesh_shape: tuple,
    axis_names: tuple,
    multichip_mode: MultichipMode,
):
    def fwd(batch, W1_block, B1_block):
        act = jnp.dot(batch, W1_block)
        act = jax.lax.psum_scatter(act, axis_names[1], scatter_dimension=1, tiled=True)
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
        use_shardy,
        multichip_mode,
        maxval=0.1,
    )
