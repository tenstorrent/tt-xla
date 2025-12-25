# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category


@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_maxpool2d():

    class MaxPool2D(torch.nn.Module):
        def __init__(self):
            super(MaxPool2D, self).__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x):
            return self.pool(x)

    run_op_test_with_random_inputs(
        MaxPool2D(),
        [(1, 64, 480, 640)],
        dtype=torch.float32,
        framework=Framework.TORCH,
    )
