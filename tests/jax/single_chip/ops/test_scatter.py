# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.lax as lax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs
from utils import Category, failed_ttmlir_compilation


# @pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.lax.scatter",
    shlo_op_name="stablehlo.scatter",
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "'ttir.scatter' op Dimension size to slice into must be 1 "
        "https://github.com/tenstorrent/tt-xla/issues/386"
    )
)
@pytest.mark.parametrize(
    "data_shape, indices_shape, updates_shape",
    [
        ((32, 32), (16, 1), (16, 32)),
        ((64, 64), (32, 1), (32, 64)),
    ],
    ids=lambda val: f"shape={val}",
)
def test_scatter(data_shape, indices_shape, updates_shape):
    def scatter(
        data: jnp.ndarray, indices: jnp.ndarray, updates: jnp.ndarray
    ) -> jnp.ndarray:
        dnums = lax.ScatterDimensionNumbers(
            update_window_dims=(1,),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )
        return lax.scatter(
            data,
            indices.astype(jnp.int32),
            updates,
            dimension_numbers=dnums,
        )

    run_op_test_with_random_inputs(
        scatter, input_shapes=[data_shape, indices_shape, updates_shape]
    )
