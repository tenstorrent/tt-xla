# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import jax
import jax.numpy as jnp
import numpy

from infrastructure import verify_module

# Note: TTNN does not support boolean data type, so bfloat16 is used instead.
# Hence the output of comparison operation is bflaot16. JAX can not perform any
# computation due to mismatch in output data type (in testing infrastructure).
# The following tests explicitly convert data type of comparison operation
# output for the verification purposes.
# [TODO] Remove this work around once the data type issue is settle.
# https://github.com/tenstorrent/tt-xla/issues/93


@pytest.mark.parametrize(
    "input_shapes",
    [[(64, 64), (64, 64)], [(128, 128), (128, 128)], [(256, 256), (256, 256)]],
)
def test_equal(input_shapes):
    def module_equal(a, b):
        c = a == b
        return jax.lax.convert_element_type(c, jnp.float32)

    verify_module(module_equal, input_shapes, dtype=jnp.bfloat16)


@pytest.mark.parametrize(
    "input_shapes",
    [[(64, 64), (64, 64)], [(128, 128), (128, 128)], [(256, 256), (256, 256)]],
)
def test_not_equal(input_shapes):
    def module_not_equal(a, b):
        c = a != b
        return jax.lax.convert_element_type(c, jnp.float32)

    verify_module(module_not_equal, input_shapes, dtype=jnp.bfloat16)


@pytest.mark.parametrize(
    "input_shapes",
    [[(64, 64), (64, 64)], [(128, 128), (128, 128)], [(256, 256), (256, 256)]],
)
def test_greater_than(input_shapes):
    def module_greater_than(a, b):
        c = a > b
        return jax.lax.convert_element_type(c, jnp.float32)

    verify_module(module_greater_than, input_shapes, dtype=jnp.bfloat16)


@pytest.mark.parametrize(
    "input_shapes",
    [[(64, 64), (64, 64)], [(128, 128), (128, 128)], [(256, 256), (256, 256)]],
)
def test_greater_equal(input_shapes):
    def module_greater_equal(a, b):
        c = a >= b
        return jax.lax.convert_element_type(c, jnp.float32)

    verify_module(module_greater_equal, input_shapes, dtype=jnp.bfloat16)


@pytest.mark.parametrize(
    "input_shapes",
    [[(64, 64), (64, 64)], [(128, 128), (128, 128)], [(256, 256), (256, 256)]],
)
def test_less_than(input_shapes):
    def module_less_than(a, b):
        c = a < b
        return jax.lax.convert_element_type(c, jnp.float32)

    verify_module(module_less_than, input_shapes, dtype=jnp.bfloat16)


@pytest.mark.parametrize(
    "input_shapes",
    [[(64, 64), (64, 64)], [(128, 128), (128, 128)], [(256, 256), (256, 256)]],
)
def test_less_equal(input_shapes):
    def module_less_equal(a, b):
        c = a <= b
        return jax.lax.convert_element_type(c, jnp.float32)

    verify_module(module_less_equal, input_shapes, dtype=jnp.bfloat16)
