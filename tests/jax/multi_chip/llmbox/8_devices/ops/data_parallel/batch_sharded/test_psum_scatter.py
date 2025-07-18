# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from infra import (
    ShardingMode,
    make_partition_spec,
    run_jax_multichip_graph_test_with_random_inputs,
)
from utils import failed_fe_compilation, failed_runtime


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
    ("x_shape", "mesh_shape", "axis_names"),
    [
        ((8192, 512), (1, 8), ("batch", "model")),
        ((8192, 512), (2, 4), ("batch", "model")),
    ],
)
@pytest.mark.parametrize(
    "sharding_mode",
    [
        pytest.param(
            ShardingMode.INPUTS_AND_MODULE,
            marks=pytest.mark.xfail(
                reason=failed_runtime(
                    "No support for rank 2 tensors in reduce scatter: "
                    "https://github.com/tenstorrent/tt-metal/issues/15010"
                )
            ),
        ),
        pytest.param(
            ShardingMode.MODULE,
            marks=pytest.mark.xfail(
                reason=failed_fe_compilation(
                    "Cannot get sharding information through the protobuf "
                    "(https://github.com/tenstorrent/tt-xla/issues/277)"
                )
            ),
        ),
    ],
)
def test_psum_scatter(
    use_shardy: bool,
    x_shape: tuple,
    mesh_shape: tuple,
    axis_names: tuple,
    sharding_mode: ShardingMode,
):
    def fwd(batch):
        act = jax.lax.psum_scatter(
            batch, axis_names[1], scatter_dimension=0, tiled=True
        )
        return act

    in_specs = (make_partition_spec(axis_names),)
    out_specs = make_partition_spec(axis_names)

    run_jax_multichip_graph_test_with_random_inputs(
        fwd,
        [x_shape],
        mesh_shape,
        axis_names,
        in_specs,
        out_specs,
        use_shardy,
        sharding_mode,
    )
