# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category



@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_conv3d():

    class Conv3D(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(
                3,
                1280,
                kernel_size=[2, 14, 14],
                stride=[2, 14, 14],
                bias=False,
            )

        def forward(self, x):
            return self.conv(x)

    run_op_test_with_random_inputs(
        Conv3D(),
        [(2204, 3, 2, 14, 14)],
        dtype=torch.float32,
        framework=Framework.TORCH,
    )
