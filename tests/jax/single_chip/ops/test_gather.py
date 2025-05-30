# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import jax.lax as lax
import pytest
from infra import run_op_test_with_random_inputs

from tests.utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.lax.gather",
    shlo_op_name="stablehlo.gather",
)
@pytest.mark.parametrize(
    "data_shape, indices_shape",
    [
        ((32, 32), (16, 1)),
        ((64, 64), (32, 1)),
    ],
    ids=lambda val: f"shape={val}",
)
def test_gather(data_shape, indices_shape):
    def gather(data: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
        # Gather needs these arguments:
        # - offset_dims: which output dims stay unchanged
        # - collapsed_slice_dims: input dims to remove after slicing
        # - start_index_map: tells which input dims the indices apply to
        dnums = lax.GatherDimensionNumbers(
            offset_dims=(1,),
            collapsed_slice_dims=(0,),
            start_index_map=(0,),
        )
        slice_sizes = (1, data.shape[1])
        return lax.gather(
            data,
            indices.astype(jnp.int32),
            dimension_numbers=dnums,
            slice_sizes=slice_sizes,
        )

    run_op_test_with_random_inputs(gather, input_shapes=[data_shape, indices_shape])
