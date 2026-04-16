# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
import torch.nn as nn
from loguru import logger

def test_softmax():

    class softmax(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input ):
            return nn.functional.softmax(input, dim=-1)


    model = softmax()
    
    input = torch.randn(1, 100, 6800, dtype=torch.bfloat16)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[input],
    )

    tester.test(workload)