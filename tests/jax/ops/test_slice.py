# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs

dim0_cases = []
for begin in jnp.arange(10).tolist():
    for end in jnp.arange(90, 100).tolist():
        dim0_cases.append((begin, end, 0))

dim1_cases = []
for begin in jnp.arange(10).tolist():
    for end in jnp.arange(90, 100).tolist():
        dim1_cases.append((begin, end, 1))

dim2_cases = []
for begin in jnp.arange(0, 64, 32).tolist():
    for end in jnp.arange(64, 128, 32).tolist():
        dim2_cases.append((begin, end, 2))

dim3_cases = []
for begin in jnp.arange(0, 64, 32).tolist():
    for end in jnp.arange(64, 128, 32).tolist():
        dim3_cases.append((begin, end, 3))


# TODO rewrite this test, this is confusing.
@pytest.mark.parametrize(
    ["begin", "end", "dim"], [*dim2_cases, *dim3_cases, *dim0_cases, *dim1_cases]
)
def test_slice(begin, end, dim):
    def module_slice(a):
        if dim == 0:
            return a[begin:end, :, :, :]
        elif dim == 1:
            return a[:, begin:end, :, :]
        elif dim == 2:
            return a[:, :, begin:end, :]
        else:
            return a[:, :, :, begin:end]

    shape = [10, 10, 10, 10]
    shape[dim] = 128

    run_op_test_with_random_inputs(module_slice, [shape])