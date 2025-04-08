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

from tests.utils import failed_fe_compilation


@pytest.mark.push
@pytest.mark.parametrize(
    "use_shardy",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    ("batch_shape", "mesh_shape", "axis_names"),
    [
        ((256, 256), (1, 4), ("batch", "model")),
    ],
)
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
def test_psum(
    use_shardy: bool,
    batch_shape: tuple,
    mesh_shape: tuple,
    axis_names: tuple,
    sharding_mode: ShardingMode,
):
    def fwd(batch):
        act = jax.lax.psum(batch, axis_names[1])
        return act

    in_specs = (make_partition_spec(axis_names),)
    out_specs = make_partition_spec((axis_names[0],))

    run_multichip_test_with_random_inputs(
        fwd,
        [batch_shape],
        mesh_shape,
        axis_names,
        in_specs,
        out_specs,
        use_shardy,
        sharding_mode,
        maxval=0.1,
    )
