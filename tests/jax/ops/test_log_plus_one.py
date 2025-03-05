# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs

from tests.utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.log1p",
    shlo_op_name="stablehlo.log_plus_one",
)
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)], ids=lambda val: f"{val}")
def test_log1p(x_shape: tuple):
    def log1p(x: jax.Array) -> jax.Array:
        return jnp.log1p(x)

    run_op_test_with_random_inputs(log1p, [x_shape])
