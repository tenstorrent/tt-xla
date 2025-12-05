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
