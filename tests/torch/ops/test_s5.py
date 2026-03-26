# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger

def test_sanity():

    class sanity(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, feature_map_size, anchor_range):
            
            device = feature_map_size.device
            
            z_centers = torch.linspace(
                anchor_range[2], anchor_range[5], 1 + 1, device=device
            )

            return  z_centers



    model = sanity()
    
    feature_map_size = torch.tensor([248, 216],dtype=torch.int64)
    anchor_range = torch.tensor([  0.00000, -39.68000,  -0.60000,  69.12000,  39.68000,  -0.60000],dtype=torch.float32)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[feature_map_size,anchor_range],
    )

    tester.test(workload)