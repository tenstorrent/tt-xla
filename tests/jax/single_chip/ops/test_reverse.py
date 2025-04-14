# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import random_tensor, run_op_test

from tests.utils import Category, failed_ttmlir_compilation


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.flip",
    shlo_op_name="stablehlo.reverse",
)
@pytest.mark.parametrize(
    ["shape"],
    [
        [(32, 32)],
        [(64, 64)],
    ],
    ids=lambda val: f"{val}",
)
@pytest.mark.skip(
    reason=failed_ttmlir_compilation(
        "failed to legalize operation 'ttir.reverse "
        "https://github.com/tenstorrent/tt-xla/issues/503"
    )
)
def test_reverse(shape: tuple):
    def reverse(a: jax.Array) -> jax.Array:
        return jnp.flip(a)

    input = random_tensor(shape, jnp.int32, minval=0, maxval=10, random_seed=3)
    run_op_test(reverse, [input])
