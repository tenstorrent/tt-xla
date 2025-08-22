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
    jax_op_name="jax.numpy.sin",
    shlo_op_name="stablehlo.sine",
)
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)], ids=lambda val: f"{val}")
def test_sin(x_shape: tuple):
    def sin(x: jax.Array) -> jax.Array:
        return jnp.sin(x)

    run_op_test_with_random_inputs(sin, [x_shape])


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.sin",
    shlo_op_name="stablehlo.sine",
)
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)], ids=lambda val: f"{val}")
@pytest.mark.parametrize("format", ["bfloat16", "bfp8"])
def test_sin_lower_df(x_shape: tuple, format: str):
    def sin(x: jax.Array) -> jax.Array:
        return jnp.sin(x)

    if format == "bfloat16":
        compiler_config = CompilerConfig()
    else:  # bfp8
        compiler_config = CompilerConfig(enable_bfp8_conversion=True)
    
    run_op_test_with_random_inputs(sin, [x_shape], dtype=jnp.bfloat16, compiler_config=compiler_config)
