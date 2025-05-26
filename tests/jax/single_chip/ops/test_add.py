# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs
import gc
import ctypes

from tests.utils import Category
import sys
import inspect


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.add",
    shlo_op_name="stablehlo.add",
)
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(10000, 10000), (10000, 10000)],
    ],
    ids=lambda val: f"{val}",
)
def test_add(x_shape: tuple, y_shape: tuple):
    def add(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.add(x, y)
    for i in range(10000, 10001):
        run_op_test_with_random_inputs(add, [(i,i), (i,i)])
        gc.collect()  # Force garbage collection
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
