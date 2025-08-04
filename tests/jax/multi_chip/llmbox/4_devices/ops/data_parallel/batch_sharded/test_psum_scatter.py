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


def conditionally_skip(x_shape: tuple, mesh_shape: tuple):
    """
    Helper function which checks test input combinations and xfails if necessary.

    Extracted here in order not to pollute the test function.
    """
    if len(x_shape) < 4:
        pytest.xfail(
            failed_runtime(
                "No support for rank 2 tensors in reduce scatter: "
                "https://github.com/tenstorrent/tt-metal/issues/15010"
            )
        )
    if mesh_shape[0] == 1 and mesh_shape[1] == 8:
        pytest.skip(failed_runtime("Floating point exception"))


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
        ((8192, 512), (1, 4), ("batch", "model")),
        ((1, 1, 8192, 512), (1, 4), ("batch", "model")),
    ],
)
# Cannot use ShardingMode.INPUTS because it does not define axis names and we are using jax.lax.psum_scatter
@pytest.mark.parametrize(
    "sharding_mode",
    [
        ShardingMode.INPUTS_AND_MODULE,
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
    conditionally_skip(x_shape, mesh_shape)

    def fwd(batch):
        act = jax.lax.psum_scatter(
            batch, axis_names[1], scatter_dimension=len(x_shape) - 1, tiled=True
        )
        return act

    partition_spec = (None,) * (len(x_shape) - 2) + axis_names
    in_specs = (make_partition_spec(partition_spec),)
    out_specs = make_partition_spec(partition_spec)

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
