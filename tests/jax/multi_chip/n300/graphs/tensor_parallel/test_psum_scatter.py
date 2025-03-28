# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from infra import make_partition_spec, run_multichip_test_with_random_inputs
import jax
import jax.numpy as jnp
import pytest
from infra import (
    make_partition_spec,
    ShardingMode,
    run_multichip_test_with_random_inputs,
)

from tests.utils import failed_ttmlir_compilation


@pytest.mark.nightly
@pytest.mark.push
@pytest.mark.parametrize(
    "use_shardy",
    [
        True,
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
    "sharding_mode",
    [
        ShardingMode.INPUTS_AND_MODULE,
        ShardingMode.MODULE,
        ShardingMode.INPUTS,
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
    sharding_mode: ShardingMode,
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
        sharding_mode,
        maxval=0.1,
    )
