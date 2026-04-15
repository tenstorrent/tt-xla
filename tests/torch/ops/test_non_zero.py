# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger

def test_nonzero():

    class nonzero(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x_mask ):
            non_valid_idx = (1 - x_mask).nonzero(as_tuple=False)
            return non_valid_idx


    model = nonzero()
    
    x_mask = torch.load("x_mask.pt",map_location="cpu")
    logger.info("x_mask={}",x_mask)
    logger.info("x_mask.shape={}",x_mask.shape)
    logger.info("x_mask.dtype={}",x_mask.dtype)


    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[x_mask],
    )

    tester.test(workload)