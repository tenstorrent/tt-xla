# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category


@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_convtranspose2d():

    class ConvTranspose2D(torch.nn.Module):
        def __init__(self, in_channels, skip_channels, out_channels):
            super(ConvTranspose2D, self).__init__()
            self.upsample = torch.nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2, padding=0
            )

        def forward(self, x):
            return self.upsample(x)

    run_op_test_with_random_inputs(
        ConvTranspose2D(512, 256, 256),
        [(1, 512, 64, 64)],
        dtype=torch.float32,
        framework=Framework.TORCH,
    )
