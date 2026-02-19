# 

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger

def test_sort():

    class sort(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, sort_ip):
            return sort_ip.sort(descending=True)[1]

    # model
    model = sort()
    model.eval()

    # inputs
    sort_ip = torch.load('sort_ip.pt',map_location="cpu")
    
    logger.info("sort_ip={}",sort_ip)
    logger.info("sort_ip.shape={}",sort_ip.shape)
    logger.info("sort_ip.dtype={}",sort_ip.dtype)
    
    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[sort_ip],
    )

    tester.test(workload)