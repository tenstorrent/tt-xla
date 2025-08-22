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
    jax_op_name="jax.numpy.sqrt",
    shlo_op_name="stablehlo.sqrt",
)
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)], ids=lambda val: f"{val}")
def test_sqrt(x_shape: tuple):
    def sqrt(x: jax.Array) -> jax.Array:
        return jnp.sqrt(x)

    # Input must be strictly positive because of sqrt(x).
    run_op_test_with_random_inputs(sqrt, [x_shape], minval=0.1, maxval=10.0)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.sqrt",
    shlo_op_name="stablehlo.sqrt",
)
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)], ids=lambda val: f"{val}")
@pytest.mark.parametrize("format", ["bfloat16", "bfp8"])
def test_sqrt_lower_df(x_shape: tuple, format: str):
    def sqrt(x: jax.Array) -> jax.Array:
        return jnp.sqrt(x)

    if format == "bfloat16":
        compiler_config = CompilerConfig()
    else:  # bfp8
        compiler_config = CompilerConfig(enable_bfp8_conversion=True)
    
    # Input must be strictly positive because of sqrt(x).
    run_op_test_with_random_inputs(sqrt, [x_shape], minval=0.1, maxval=10.0, dtype=jnp.bfloat16, compiler_config=compiler_config)
