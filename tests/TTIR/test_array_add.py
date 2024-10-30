# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from infrastructure import verify_test_module


@verify_test_module([(2, 2), (2, 2)], [(3, 2), (3, 2)])
def test_add(a, b):
    return a + b


@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5, 6])
def test_module_add(rank):
    input_shape = (32,) if rank == 1 else (1,) * (rank - 2) + (32, 32)

    @verify_test_module([input_shape, input_shape])
    def module_add(a, b):
        c = a + a
        d = b + b
        return c + d

    module_add()
