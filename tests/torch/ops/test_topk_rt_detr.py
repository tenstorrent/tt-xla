# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger

torch.set_printoptions(
    threshold=torch.inf,  # print all elements
    precision=6,          # decimal places
    linewidth=200,        # characters per line before wrapping
)

def test_topk():

    class topk(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inputs):
            _, topk_ind = torch.topk(inputs, 300, dim=1)
            return topk_ind


    model = topk().to(torch.bfloat16)
    model.eval()
    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)
    
    inputs = torch.load('topk_ip.pt',map_location="cpu")
    
    logger.info("inputs={}",inputs)
    logger.info("inputs.shape={}",inputs.shape)
    logger.info("inputs.dtype={}",inputs.dtype)

    # workload = Workload(
    #     framework=Framework.TORCH,
    #     model=model,
    #     args=[inputs],
    # )

    # tester.test(workload)