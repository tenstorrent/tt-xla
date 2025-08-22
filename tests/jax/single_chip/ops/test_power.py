# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs
from infra.compiler_config import CompilerConfig
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.power",
    shlo_op_name="stablehlo.power",
)
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
    ids=lambda val: f"{val}",
)
def test_power(x_shape: tuple, y_shape: tuple):
    def power(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.power(x, y)

    run_op_test_with_random_inputs(power, [x_shape, y_shape])


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.power",
    shlo_op_name="stablehlo.power",
)
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
    ids=lambda val: f"{val}",
)
@pytest.mark.parametrize("format", ["bfloat16", "bfp8"])
def test_power_lower_df(x_shape: tuple, y_shape: tuple, format: str):
    def power(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.power(x, y)

    if format == "bfloat16":
        compiler_config = CompilerConfig()
    else:  # bfp8
        compiler_config = CompilerConfig(enable_bfp8_conversion=True)
    
    run_op_test_with_random_inputs(power, [x_shape, y_shape], dtype=jnp.bfloat16, compiler_config=compiler_config)
