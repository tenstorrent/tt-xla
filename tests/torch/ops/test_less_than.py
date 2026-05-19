# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger

def test_lt():

    class lt(torch.nn.Module):
        def forward(self, anchors ):
            return anchors < 0.99

    model = lt()

    ip = torch.load("anchors.pt",map_location="cpu")

    logger.info("ip={}",ip)
    logger.info("ip.shape={}",ip.shape)
    logger.info("ip.dtype={}",ip.dtype)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[ip],
    )

    tester.test(workload)