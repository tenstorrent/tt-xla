# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import run_op_test_with_random_inputs
from utils import Category

dim0_cases = []
for begin in range(10):
    for end in range(90, 100):
        dim0_cases.append((begin, end, 0))

dim1_cases = []
for begin in range(10):
    for end in range(90, 100):
        dim1_cases.append((begin, end, 1))

dim2_cases = []
for begin in range(0, 64, 32):
    for end in range(64, 128, 32):
        dim2_cases.append((begin, end, 2))

dim3_cases = []
for begin in range(0, 64, 32):
    for end in range(64, 128, 32):
        dim3_cases.append((begin, end, 3))


# TODO investigate if this test can be rewritten to make it easier for understanding.
# @pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.lax.slice",
    shlo_op_name="stablehlo.slice",
)
@pytest.mark.parametrize(
    ["begin", "end", "dim"],
    [*dim2_cases, *dim3_cases, *dim0_cases, *dim1_cases],
    ids=lambda val: f"{val}",
)
def test_slice(begin: int, end: int, dim: int):
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
