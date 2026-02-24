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
            super(Conv3D, self).__init__()
            self.patch_size = 16
            self.temporal_patch_size = 2
            self.in_channels = 3
            self.embed_dim = 1024

            kernel_size = [
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ]

            self.conv = torch.nn.Conv3d(
                self.in_channels,
                self.embed_dim,
                kernel_size=kernel_size,
                stride=kernel_size,
                bias=True,
            )

        def forward(self, x):
            return self.conv(x)

    run_op_test_with_random_inputs(
        Conv3D().to(torch.bfloat16).eval(),
        [(11008, 3, 2, 16, 16)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
    )
