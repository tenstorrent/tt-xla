# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import (
    ShardingMode,
    make_partition_spec,
    run_jax_multichip_graph_test_with_random_inputs,
)
from infra.evaluators import ComparisonConfig, PccConfig
from utils import failed_fe_compilation


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
        ((8192, 784), (784, 2048), (2048,), (1, 8), ("batch", "model")),
        ((8192, 784), (784, 2048), (2048,), (2, 4), ("batch", "model")),
    ],
)
# Cannot use ShardingMode.INPUTS because it does not define axis names and we are using jax.lax.psum
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
def test_dot_psum(
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
        act = jax.lax.psum(act, axis_names[1])
        act = act + B1_block
        return act

    in_specs = (
        make_partition_spec(axis_names),
        make_partition_spec((axis_names[1], None)),
        make_partition_spec((None,)),
    )
    out_specs = make_partition_spec((axis_names[0],))

    run_jax_multichip_graph_test_with_random_inputs(
        fwd,
        [batch_shape, W1_shape, B1_shape],
        mesh_shape,
        axis_names,
        in_specs,
        out_specs,
        use_shardy,
        sharding_mode,
        maxval=0.1,
        comparison_config=ComparisonConfig(
            pcc=PccConfig(required_pcc=0.88)
        ),  # https://github.com/tenstorrent/tt-xla/issues/1161
    )
