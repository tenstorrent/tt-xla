# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category
from loguru import logger


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_case1():

    class Case1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=360,
                out_channels=480,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                groups=1,
            )

        def forward(self, x):
            x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
            return self.conv(x)
    
    model = Case1().to(torch.bfloat16)
    
    logger.info("model={}",model)
    
    run_op_test_with_random_inputs(
        model,
        [(1, 360, 28, 40)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
    )
    
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_case2():

    class Case2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=360,
                out_channels=480,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                groups=1,
            )

        def forward(self, x):
            x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
            return x
    
    model = Case2().to(torch.bfloat16)
    
    logger.info("model={}",model)
    
    run_op_test_with_random_inputs(
        model,
        [(1, 360, 28, 40)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
    )

@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_case3():

    class Case3(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=360,
                out_channels=480,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                groups=1,
            )

        def forward(self, x):
            return  self.conv(x)
    
    model = Case3().to(torch.bfloat16)
    
    logger.info("model={}",model)
    
    # avgpool2d on cpu
    x = torch.randn((1, 360, 28, 40))
    x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
    
    logger.info("x.shape={}",x.shape)
    
    run_op_test_with_random_inputs(
        model,
        [x.shape],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
    )