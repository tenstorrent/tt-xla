# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.nn.MaxPool2d",
)
def test_max_pool2d():
    """Test max_pool2d operation."""

    class MaxPool2d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        def forward(self, input_tensor):
            return self.maxpool(input_tensor)

    model = MaxPool2d()

    input_shape = (6, 64, 464, 800)

    run_op_test_with_random_inputs(
        model, [input_shape], framework=Framework.TORCH
    )