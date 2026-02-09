# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from utils import Category

from loguru import logger

# torch.set_printoptions(threshold=float('inf'))

def test_s():

    class sanity(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, anchors):
            output = anchors < 0.99
            
            logger.info("output={}",output)
            logger.info("output.dtype={}",output.dtype)
            logger.info("output.shape={}",output.shape)
            
            return output

    model = sanity().to(torch.bfloat16)
    model.eval()
    anchors = torch.load("anchors.pt",map_location="cpu")
    
    logger.info("anchors={}",anchors)
    logger.info("anchors.shape={}",anchors.shape)
    logger.info("anchors.dtype={}",anchors.dtype)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[anchors],
    )

    tester.test(workload)