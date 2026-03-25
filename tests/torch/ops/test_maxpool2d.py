# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger

def test_maxpool2d():

    class maxpool2d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x ):
            return F.max_pool2d(
                x,
                (3, 3),
                (2, 2),
                (0, 0),
                (1, 1),
                False,
            )


    model = maxpool2d().to(torch.bfloat16)
    model.eval()
    
    inputs = torch.load('maxpool2d_new_ip.pt',map_location="cpu")
    logger.info("inputs={}",inputs)
    logger.info("inputs.shape={}",inputs.shape)
    logger.info("inputs.dtype={}",inputs.dtype)
        
    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[inputs],
    )

    tester.test(workload)