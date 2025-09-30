# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.lax as lax
import jax.numpy as jnp
import pytest
from infra import ComparisonConfig, run_op_test_with_random_inputs, run_op_test
from utils import Category, failed_ttmlir_compilation


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.lax.scatter",
    shlo_op_name="stablehlo.scatter",
)
@pytest.mark.parametrize(
    "data_shape, indices_shape, updates_shape",
    [
        ((32, 32), (16, 1), (16, 32)),
        ((64, 64), (32, 1), (32, 64)),
    ],
    ids=lambda val: f"shape={val}",
)
def test_scatter_1(data_shape, indices_shape, updates_shape):
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

    data = jnp.arange(jnp.prod(jnp.array(data_shape)), dtype=jnp.int32).reshape(
        data_shape
    )
    indices = jnp.arange(jnp.prod(jnp.array(indices_shape)), dtype=jnp.int32).reshape(
        indices_shape
    )
    updates = (
        jnp.arange(jnp.prod(jnp.array(updates_shape)), dtype=jnp.int32).reshape(
            updates_shape
        )
        + 1000
    )

    comparison_config = ComparisonConfig()
    comparison_config.equal.enable()

    run_op_test(
        scatter,
        inputs=[data, indices, updates],
        comparison_config=comparison_config,
    )


@pytest.mark.parametrize(
    "data_shape, indices_shape, updates_shape",
    [
        ((1, 3, 320, 320), (1, 1), (1, 3, 320, 320)),
    ],
    ids=lambda val: f"shape={val}",
)
def test_scatter_2(data_shape, indices_shape, updates_shape):
    def scatter(
        data: jnp.ndarray, indices: jnp.ndarray, updates: jnp.ndarray
    ) -> jnp.ndarray:
        dnums = lax.ScatterDimensionNumbers(
            update_window_dims=(1, 2, 3),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )
        return lax.scatter(
            data,
            indices.astype(jnp.int32),
            updates,
            dimension_numbers=dnums,
        )

    data = jnp.arange(jnp.prod(jnp.array(data_shape)), dtype=jnp.int32).reshape(
        data_shape
    )
    indices = jnp.array([0], dtype=jnp.int32).reshape(indices_shape)
    updates = (
        jnp.arange(jnp.prod(jnp.array(updates_shape)), dtype=jnp.int32).reshape(
            updates_shape
        )
        + 100000
    )  # offset to avoid overlap with data

    comparison_config = ComparisonConfig()
    comparison_config.equal.enable()

    run_op_test(
        scatter,
        inputs=[data, indices, updates],
        comparison_config=comparison_config,
    )


# gpt-oss:
@pytest.mark.parametrize(
    "data_shape, indices_shape, updates_shape",
    [
        ((71, 32), (71, 4, 2), (71, 4)),
    ],
    ids=lambda val: f"shape={val}",
)
def test_scatter_3(data_shape, indices_shape, updates_shape):
    def scatter(
        data: jnp.ndarray, indices: jnp.ndarray, updates: jnp.ndarray
    ) -> jnp.ndarray:
        dnums = lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0, 1),
            scatter_dims_to_operand_dims=(0, 1),
        )
        return lax.scatter(
            data,
            indices.astype(jnp.int32),
            updates,
            dimension_numbers=dnums,
        )

    data = jnp.arange(jnp.prod(jnp.array(data_shape)), dtype=jnp.float32).reshape(
        data_shape
    )
    col_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    row_indices = (
        jnp.arange(data_shape[0], dtype=jnp.int32).reshape(-1, 1).repeat(4, axis=1)
    )
    indices = jnp.stack(
        [row_indices, col_indices.reshape(1, -1).repeat(data_shape[0], axis=0)], axis=-1
    )

    updates = (
        jnp.arange(jnp.prod(jnp.array(updates_shape)), dtype=jnp.float32).reshape(
            updates_shape
        )
        + 50000
    )

    run_op_test(
        scatter,
        inputs=[data, indices, updates],
    )


@pytest.mark.parametrize(
    "data_shape, indices_shape, updates_shape",
    [
        ((1000, 32), (10, 1), (10, 32)),
    ],
    ids=lambda val: f"shape={val}",
)
def test_scatter_4(data_shape, indices_shape, updates_shape):
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
        scatter,
        input_shapes=[data_shape, indices_shape, updates_shape],
        minval=0,
        maxval=data_shape[0],
    )
