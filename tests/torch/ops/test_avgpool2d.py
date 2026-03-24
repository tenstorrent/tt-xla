# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger
import torch.nn.functional as F

def test_avgpool2d():

    class avgpool2d(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x ):
            return F.avg_pool2d(x, kernel_size=2, stride=2)


    model = avgpool2d().to(torch.bfloat16)
    model.eval()
    
    inputs = torch.load('avgpool2d_ip.pt',map_location="cpu")
    
    logger.info("inputs={}",inputs)
    logger.info("inputs.shape={}",inputs.shape)
    logger.info("inputs.dtype={}",inputs.dtype)
    logger.info("model={}",model)
    
    
    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[inputs],
    )

    tester.test(workload)