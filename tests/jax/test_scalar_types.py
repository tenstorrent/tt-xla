# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from infra import run_op_test_with_random_inputs

# TODO not sure what this is supposed to test. Rethink.


@pytest.mark.parametrize("x_shape", [(3, 3)])
def test_scalar_type(x_shape: tuple):
    def scalar_type(x: jax.Array):
        return x.shape[0]

    run_op_test_with_random_inputs(scalar_type, [x_shape])
