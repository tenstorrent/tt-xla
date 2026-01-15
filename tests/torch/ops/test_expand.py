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
    "input_shape,expand_sizes",
    [
        # 1D tensor expansion
        ((1,), (10,)),
        ((5,), (-1,)),
        # 2D expansion - expanding singleton dimensions
        ((1, 5), (10, 5)),
        ((5, 1), (5, 10)),
        ((1, 1), (10, 10)),
        # Using -1 to keep dimension unchanged
        ((3, 1), (-1, 4)),
        ((1, 5), (10, -1)),
        # 3D tensor expansion
        ((1, 4, 5), (3, 4, 5)),
        ((3, 1, 5), (3, 4, 5)),
        ((3, 4, 1), (3, 4, 5)),
        ((1, 1, 5), (3, 4, 5)),
        ((1, 4, 1), (3, 4, 5)),
        ((3, 1, 1), (3, 4, 5)),
        ((1, 1, 1), (3, 4, 5)),
        # Using -1 to keep dimension unchanged
        ((4, 1, 8), (4, 5, -1)),
        ((1, 1, 1), (-1, 10, 20)),
        # 4D tensor expansion
        ((1, 3, 1, 1), (8, 3, 32, 32)),
        ((1, 1, 224, 224), (16, 3, 224, 224)),
        ((2, 1, 4, 1), (2, 5, 4, 10)),
        # Dimension addition (expanding to more dimensions)
        ((3, 4), (2, 3, 4)),
        ((5,), (3, 5)),
        ((2, 3), (4, 5, 2, 3)),
        ((1, 1), (2, 3, 1, 1)),
        # Large expansions
        ((1, 10), (100, 10)),
        ((1, 1), (100, 100)),
        ((1, 5, 1), (50, 5, 20)),
        # No expansion (identity cases)
        ((5, 10), (5, 10)),
        ((3, 4, 5), (3, 4, 5)),
        # Edge cases with size 1
        ((1,), (1,)),  # Single element, no expansion
        ((1, 1, 1, 1), (1, 1, 1, 1)),  # All singleton, no expansion
        # Complex -1 patterns
        ((1, 5, 1, 10), (-1, -1, 20, -1)),  # Multiple -1 with expansion
        ((1, 1, 7), (3, 4, -1)),  # -1 on non-singleton dimension
    ],
)
def test_expand_parameterized(input_shape, expand_sizes):
    """Test expand operation with various input shapes and expansion sizes"""

    class Expand(torch.nn.Module):
        def __init__(self, expand_sizes):
            super().__init__()
            self.expand_sizes = expand_sizes

        def forward(self, x):
            return x.expand(*self.expand_sizes)

    expand = Expand(expand_sizes)

    run_op_test_with_random_inputs(
        expand, [input_shape], dtype=torch.bfloat16, framework=Framework.TORCH
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_expand_different_dtypes(dtype):
    """Test expand operation with different data types"""

    class Expand(torch.nn.Module):
        def forward(self, x):
            return x.expand(10, 20)

    expand = Expand()

    run_op_test_with_random_inputs(
        expand, [(1, 20)], dtype=dtype, framework=Framework.TORCH
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_expand_gpt_oss_mlp():
    class Expand(torch.nn.Module):
        def forward(self, x):
            return x.unsqueeze(0).expand(32, -1, -1).flatten(0, 1)

    expand = Expand()

    run_op_test_with_random_inputs(
        expand, [(128, 2880)], dtype=torch.bfloat16, framework=Framework.TORCH
    )
