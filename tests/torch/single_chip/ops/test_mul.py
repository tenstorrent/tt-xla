# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from infra import Framework, run_op_test, run_op_test_with_random_inputs
from infra.comparators.torch_comparator import TorchComparator
from utils import Category

from tests.infra.comparators.comparison_config import ComparisonConfig


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_mul():
    class Mul(torch.nn.Module):
        def forward(self, x, y):
            return x * y

    mul = Mul()

    run_op_test_with_random_inputs(
        mul, [(32, 32), (32, 32)], dtype=torch.bfloat16, framework=Framework.TORCH
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_numpy_inplace_multiply_torch_override():
    """
    Test that TorchFunctionOverride works correctly for numpy array inplace multiplication operations.
    """
    from tt_torch.torch_overrides import torch_function_override

    def inplace_multiply(T):
        T[:2, :] *= 8
        return T

    input_T = np.array([[0, -1, 16], [-1, 0, 32], [0, 0, 1]], dtype=np.float32)

    # Temporarily disable the override to get golden output
    torch_function_override.__exit__(None, None, None)
    try:
        golden_compiled_op = torch.compile(inplace_multiply)
        golden = golden_compiled_op(input_T)
        golden = torch.from_numpy(golden)
    finally:
        torch_function_override.__enter__()  # Always re-enable it

    compiled_op = torch.compile(inplace_multiply)
    output = compiled_op(input_T)
    output = torch.from_numpy(output)

    comparator = TorchComparator(ComparisonConfig())
    comparator.compare(output, golden)
