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

        def forward(self, feature_map_size, anchor_range, rotations ):
            
            device = feature_map_size.device
            
            x_centers = torch.linspace(
                anchor_range[0], anchor_range[3], feature_map_size[1] + 1, device=device
            ).to(torch.float32)
            y_centers = torch.linspace(
                anchor_range[1], anchor_range[4], feature_map_size[0] + 1, device=device
            ).to(torch.float32)
            z_centers = torch.linspace(
                anchor_range[2], anchor_range[5], 1 + 1, device=device).to(torch.float32)

            x_shift = (x_centers[1] - x_centers[0]) / 2
            y_shift = (y_centers[1] - y_centers[0]) / 2
            z_shift = (z_centers[1] - z_centers[0]) / 2
            x_centers = (
                x_centers[: feature_map_size[1]] + x_shift
            )  
            y_centers = (
                y_centers[: feature_map_size[0]] + y_shift
            )  
            z_centers = z_centers[:1] + z_shift  
          
            meshgrids = torch.meshgrid(x_centers, y_centers, z_centers, rotations)

            return meshgrids


    model = sanity()
    
    feature_map_size = torch.tensor([248, 216],dtype=torch.int64)
    anchor_range = torch.tensor([  0.00000, -39.68000,  -0.60000,  69.12000,  39.68000,  -0.60000],dtype=torch.float32)
    rotations = torch.tensor([0.00000, 1.57000],dtype=torch.float32)
    

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[feature_map_size,anchor_range,rotations],
    )

    tester.test(workload)