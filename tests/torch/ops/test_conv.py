# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger
from torchvision import models

def test_sanity():

    
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.conv = model.features[0][0] 

        def forward(self, ip):
            x = self.conv(ip)
            return x
        
        
    model = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)
    model.eval()
    model = Wrapper(model).to(torch.bfloat16)
    
    inputs = torch.load('conv_ip.pt')
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