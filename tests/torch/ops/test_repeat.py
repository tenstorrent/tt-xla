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
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize(
    "input_shape,repeat_dims",
    [
        # 1D tensor tests
        ((10,), (3,)),
        ((100,), (1,)),
        ((50,), (5,)),
        # 2D tensor tests - repeat along different dimensions
        ((32, 64), (1, 1)),  # No repeat (identity)
        ((32, 64), (2, 1)),  # Repeat rows only
        ((32, 64), (1, 3)),  # Repeat columns only
        ((32, 64), (2, 3)),  # Repeat both dimensions
        ((1, 100), (10, 1)),  # Single row repeated
        ((100, 1), (1, 10)),  # Single column repeated
        # 3D tensor tests
        ((4, 8, 16), (1, 1, 1)),  # No repeat
        ((4, 8, 16), (2, 1, 1)),  # Repeat first dimension
        ((4, 8, 16), (1, 2, 1)),  # Repeat middle dimension
        ((4, 8, 16), (1, 1, 2)),  # Repeat last dimension
        ((4, 8, 16), (2, 2, 2)),  # Repeat all dimensions
        ((2, 3, 4), (3, 2, 1)),  # Mixed repeats
        # 4D tensor tests
        ((2, 3, 4, 5), (1, 1, 1, 1)),  # No repeat
        ((2, 3, 4, 5), (2, 1, 1, 1)),  # Repeat batch dimension
        ((2, 3, 4, 5), (1, 2, 3, 4)),  # Different repeats per dimension
        # Edge cases
        ((1,), (100,)),  # Single element repeated many times
        ((1, 1), (10, 10)),  # Single element in 2D
        ((100, 100), (1, 1)),  # Large tensor, no repeat
        # Broadcasting-like patterns
        (
            (128, 2880),
            (32, 1),
        ),  # GPT-OSS MOE - should be decomposed into unsqueeze + expand + flatten
        ((5, 1), (1, 10)),  # Expand along second dimension
        ((1, 5), (10, 1)),  # Expand along first dimension
    ],
)
def test_repeat_parameterized(input_shape, repeat_dims):
    class Repeat(torch.nn.Module):
        def __init__(self, repeat_dims):
            super().__init__()
            self.repeat_dims = repeat_dims

        def forward(self, x):
            return x.repeat(*self.repeat_dims)

    repeat = Repeat(repeat_dims)

    run_op_test_with_random_inputs(
        repeat, [input_shape], dtype=torch.bfloat16, framework=Framework.TORCH
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_repeat_different_dtypes(dtype):
    """Test repeat operation with different data types"""

    class Repeat(torch.nn.Module):
        def forward(self, x):
            return x.repeat(2, 3)

    repeat = Repeat()

    run_op_test_with_random_inputs(
        repeat, [(16, 32)], dtype=dtype, framework=Framework.TORCH
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_repeat_dimension_expansion():
    """Test repeating with more dimensions than input (adds new dimensions)"""

    class Repeat(torch.nn.Module):
        def forward(self, x):
            return x.repeat(4, 2, 1)

    repeat = Repeat()

    run_op_test_with_random_inputs(
        repeat, [(10, 20)], dtype=torch.bfloat16, framework=Framework.TORCH
    )
