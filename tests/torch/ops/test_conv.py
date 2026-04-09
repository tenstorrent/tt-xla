# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger

def test_conv():

    class AtrousConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(18, 18),
                dilation=(18, 18),
                bias=False,
                groups=1,
            )

        def forward(self, x):
            return self.conv(x)

    model = AtrousConv()

    x = torch.randn(6, 512, 16, 44, dtype=torch.float32)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[x],
    )

    tester.test(workload)
