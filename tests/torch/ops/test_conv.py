# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger

def test_conv():

    class conv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(
                128, 128, kernel_size=3, stride=2, padding=1, bias=False
            )

        def forward(self, x):
            return self.conv2d(x)


    model = conv()

    x = torch.randn(6, 128, 120, 200, dtype=torch.float32)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[x],
    )

    tester.test(workload)