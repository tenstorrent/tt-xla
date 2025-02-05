# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from infra import run_graph_test_with_random_inputs


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)])
def test_simple_gradient(x_shape: tuple):
    def simple_gradient(x: jax.Array):
        def gradient(x: jax.Array):
            return (x**2).sum()

        return jax.grad(gradient)(x)

    run_graph_test_with_random_inputs(simple_gradient, [x_shape])
