# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger

def test_sigmoid():

    class sigmoid(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self,  inputs):

            return torch.sigmoid(inputs)


    model = sigmoid()
    model.eval()
    
    inputs = torch.load('sig_ip.pt',map_location="cpu")
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