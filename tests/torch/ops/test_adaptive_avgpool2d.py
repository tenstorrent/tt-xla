# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger
import torch.nn as nn

def test_adaptive_avgpool2d_case1():

    class adaptive_avgpool2d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.avgpool_img = nn.AdaptiveAvgPool2d((5, 22))

        def forward(self, img_features ):
            
            output = self.avgpool_img(img_features)

            return output


    model = adaptive_avgpool2d().to(torch.bfloat16)
    
    image_features = torch.load('image_features.pt',map_location="cpu")
    
    logger.info("image_features={}",image_features)
    logger.info("image_features.shape={}",image_features.shape)
    logger.info("image_features.dtype={}",image_features.dtype)


    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[image_features],
    )

    tester.test(workload)
    

def test_adaptive_avgpool2d_case2():

    class adaptive_avgpool2d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.avgpool_lidar = nn.AdaptiveAvgPool2d((8, 8))

        def forward(self, lidar_features  ):
            
            output = self.avgpool_lidar(lidar_features)

            return output


    model = adaptive_avgpool2d().to(torch.bfloat16)
    
    lidar_features = torch.load('lidar_features.pt',map_location="cpu")
    
    logger.info("lidar_features={}",lidar_features)
    logger.info("lidar_features.shape={}",lidar_features.shape)
    logger.info("lidar_features.dtype={}",lidar_features.dtype)
    

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[lidar_features],
    )

    tester.test(workload)