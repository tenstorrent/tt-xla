# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.lax as lax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs
from utils import Category


def conditionally_skip(start_index_map: tuple):
    """
    Helper function which checks test input combinations and xfails if necessary.

    Extracted here in order not to pollute the test function.
    """
    if len(start_index_map) > 1:
        pytest.xfail("Test cases that index multiple dimensions need integer inputs")


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.lax.gather",
    shlo_op_name="stablehlo.gather",
)
@pytest.mark.parametrize(
    "data_shape, indices_shape, offset_dims, start_index_map, slice_sizes",
    [
        ((2048, 1, 200), (1, 6, 1), (2, 3), (0,), (1, 1, 200)),
        ((32000, 3200), (1, 6, 1), (2,), (0,), (1, 3200)),
        ((8, 26, 26, 192), (1,), (0, 1, 2), (3,), (8, 26, 26, 1)),
        ((8, 54, 54, 64), (1,), (0, 1, 2), (3,), (8, 54, 54, 1)),
        ((2, 7, 512), (2, 2), (1,), (0, 1), (1, 1, 512)),
        ((2, 7, 768), (2, 2), (1,), (0, 1), (1, 1, 768)),
    ],
    ids=lambda val: f"shape={val}",
)
def test_gather(data_shape, indices_shape, offset_dims, start_index_map, slice_sizes):
    conditionally_skip(start_index_map)

    def gather(data: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
        # Gather needs these arguments:
        # - offset_dims: which output dims stay unchanged
        # - collapsed_slice_dims: input dims to remove after slicing
        # - start_index_map: tells which input dims the indices apply to
        dnums = lax.GatherDimensionNumbers(
            offset_dims=offset_dims,
            collapsed_slice_dims=start_index_map,
            start_index_map=start_index_map,
        )
        return lax.gather(
            data,
            indices.astype(jnp.int32),
            dimension_numbers=dnums,
            slice_sizes=slice_sizes,
        )

    indexing_dim_sizes = jnp.array(data_shape)[jnp.array(start_index_map)]
    run_op_test_with_random_inputs(
        gather,
        input_shapes=[data_shape, indices_shape],
        minval=0,
        maxval=jnp.min(indexing_dim_sizes),
        dtype=jnp.int32
    )
