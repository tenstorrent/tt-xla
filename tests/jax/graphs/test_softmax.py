# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from infra import run_graph_test_with_random_inputs


@pytest.mark.parametrize(
    ["x_shape", "axis"],
    [
        [(32, 32), 0],
        [(32, 32), 1],
        [(64, 64), 0],
        [(64, 64), 1],
    ],
)
@pytest.mark.xfail(
    reason="Atol comparison failed. Calculated: atol=inf. Required: atol=0.16"
)
def test_softmax(x_shape: tuple, axis: int):
    def softmax(x: jax.Array) -> jax.Array:
        return jax.nn.softmax(x, axis=axis)

    run_graph_test_with_random_inputs(softmax, [x_shape])
