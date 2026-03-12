# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger

def test_softmax():

    class softmax(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x ):
            return torch.nn.functional.softmax(x, dim=-1)

    model = softmax().to(torch.bfloat16)
    inputs = torch.randn(1,100,6800, dtype=torch.bfloat16)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[inputs],
    )

    tester.test(workload)