# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger

def test_GN():

    class GN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=256, eps=1e-06, affine=True)

        def forward(self, hidden_states):
            return self.norm1(hidden_states)


    model = GN()
    model.eval()
    ip = torch.randn(1, 256, 1024, 1024, dtype=torch.float32)


    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[ip],
    )

    tester.test(workload)