# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax
import jax.numpy as jnp
import flax

from infrastructure import verify_module
@pytest.mark.parametrize("source_and_target_shape",
    [((8, 32, 256), (2, 4, 32, 256)),
     ((8, 32, 32), (1, 2, 4, 32, 32)),
     ((8192, 128), (1, 256, 32, 128))
     ],
    ids=["1", "2", "3"])
def test_reshape(source_and_target_shape):
    act_shape, target_shape = source_and_target_shape
    def module_reshape(act):
        return jnp.reshape(act, target_shape)

    verify_module(module_reshape, [act_shape])
