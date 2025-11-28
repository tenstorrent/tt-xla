# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_bitwise_or():
    class BitwiseOr(torch.nn.Module):
        def forward(self, x, y):
            return torch.bitwise_or(x, y)

    op = BitwiseOr()

    # Use boolean tensors for bitwise_or
    run_op_test_with_random_inputs(
        op,
        [(13,13), (13,13)],
        dtype=torch.bool,  # boolean dtype
        framework=Framework.TORCH
    )
