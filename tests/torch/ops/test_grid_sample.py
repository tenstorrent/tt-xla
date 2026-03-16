# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger
from torch import nn

def test_grid_sample():

    class GridSample(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_tensor, grid):
            
            return nn.functional.grid_sample(
                input_tensor, 
                grid, 
                mode="bilinear", 
                padding_mode="zeros", 
                align_corners=False
            )

    model = GridSample().to(torch.bfloat16)
    input_tensor = torch.load("value_l_.pt",map_location="cpu")
    grid = torch.load("sampling_grid_l_.pt",map_location="cpu")
    
    logger.info("input_tensor={}",input_tensor)
    logger.info("input_tensor.shape={}",input_tensor.shape)
    logger.info("input_tensor.dtype={}",input_tensor.dtype)
    
    logger.info("grid={}",grid)
    logger.info("grid.shape={}",grid.shape)
    logger.info("grid.dtype={}",grid.dtype)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[input_tensor,grid],
    )

    tester.test(workload)