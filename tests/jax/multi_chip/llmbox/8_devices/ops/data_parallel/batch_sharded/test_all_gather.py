# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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
        pytest.param(
            True,
            marks=pytest.mark.skip(
                reason=failed_ttmlir_compilation("Shardy does not support 1D meshes")
            ),
        ),
        False,
    ],
)
@pytest.mark.parametrize(
    ("x_shape", "mesh_shape", "axis_names"), [((8192, 784), (8,), ("batch",))]
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
def test_all_gather(
    use_shardy: bool,
    x_shape: tuple,
    mesh_shape: tuple,
    axis_names: tuple,
    sharding_mode: ShardingMode,
):
    def fwd(batch):
        act = jax.lax.all_gather(batch, axis_names, axis=0, tiled=True)
        return act

    in_specs = (make_partition_spec(axis_names),)
    out_specs = make_partition_spec(axis_names)

    run_multichip_test_with_random_inputs(
        fwd,
        [x_shape],
        mesh_shape,
        axis_names,
        in_specs,
        out_specs,
        use_shardy,
        sharding_mode,
    )
