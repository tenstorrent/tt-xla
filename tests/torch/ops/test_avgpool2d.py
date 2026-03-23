# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger
import torch.nn as nn

def test_avgpool2d():

    class avgpool2d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False)

        def forward(self, x ):
            return self.pool(x)


    model = avgpool2d().to(torch.bfloat16)
    model.eval()
    
    inputs = torch.load('avgpool2d_ip_inception.pt',map_location="cpu")
    pool = model.pool
    
    logger.info("inputs={}",inputs)
    logger.info("inputs.shape={}",inputs.shape)
    logger.info("inputs.dtype={}",inputs.dtype)
    logger.info("model.pool={}", pool)
    logger.info(
        "avgpool2d params: kernel_size={}, stride={}, padding={}, ceil_mode={}, "
        "count_include_pad={}, divisor_override={}",
        pool.kernel_size,
        pool.stride,
        pool.padding,
        pool.ceil_mode,
        pool.count_include_pad,
        pool.divisor_override,
    )

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[inputs],
    )

    tester.test(workload)