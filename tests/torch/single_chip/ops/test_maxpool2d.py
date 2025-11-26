# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category


@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.nn.MaxPool2d",
)
def test_maxpool2d():

    class MaxPool2d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.maxpool = torch.nn.MaxPool2d(
                kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False
            )

        def forward(self, x):
            return self.maxpool(x)

    model = MaxPool2d().to(torch.bfloat16)
    input_shape = (1, 256, 12, 20)

    run_op_test_with_random_inputs(
        model, [input_shape], dtype=torch.bfloat16, framework=Framework.TORCH
    )
