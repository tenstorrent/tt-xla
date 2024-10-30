# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax
import jax.numpy as jnp

from infrastructure import verify_module


def test_2x2_array_add():
    def module_add(a, b):
        return a + b

    verify_module(module_add, [(2, 2), (2, 2)])


def test_3x2_array_add():
    def module_add(a, b):
        return a + b

    verify_module(module_add, [(3, 2), (3, 2)])


@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5, 6])
def test_module_add(rank):
    def module_add(a, b):
        c = a + a
        d = b + b
        return c + d

    input_shape = []
    for i in range(rank):
        input_shape.insert(0, 32 if i < 2 else 1)

    input_shape = tuple(input_shape)
    verify_module(module_add, [input_shape, input_shape])
