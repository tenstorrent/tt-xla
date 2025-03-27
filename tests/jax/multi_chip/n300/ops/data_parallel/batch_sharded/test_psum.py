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
    ("batch_shape", "mesh_shape", "axis_names"),
    [
        ((256, 256), (1, 2), ("batch", "model")),
    ],
)
@pytest.mark.parametrize(
    "multichip_mode",
    [
        ShardingMode.INPUTS_AND_MODULE,
    ],
)
def test_psum(
    use_shardy: bool,
    batch_shape: tuple,
    mesh_shape: tuple,
    axis_names: tuple,
    multichip_mode: ShardingMode,
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
        multichip_mode,
        maxval=0.1,
    )
