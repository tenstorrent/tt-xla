# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax.numpy as jnp

from infrastructure import verify_module


@pytest.mark.parametrize(
    ["act_shape", "target_shape"],
    [
        ((8, 32, 256), (2, 4, 32, 256)),
        ((8, 32, 32), (1, 2, 4, 32, 32)),
        ((8192, 128), (1, 256, 32, 128)),
    ],
    ids=["1", "2", "3"],
)
def test_reshape(act_shape, target_shape):
    def module_reshape(act):
        return jnp.reshape(act, target_shape)

    verify_module(module_reshape, [act_shape])
