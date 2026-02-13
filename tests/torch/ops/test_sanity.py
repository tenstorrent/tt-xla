# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger

def test_sanity1():

    class s1(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, valid_mask ):

            if not valid_mask.all():
                return valid_mask
            else :
                return valid_mask
            
        
                 
        
    model = s1()
    
    valid_mask = torch.load('valid_mask.pt',map_location="cpu")
    logger.info("valid_mask={}",valid_mask)
    logger.info("valid_mask.shape={}",valid_mask.shape)
    logger.info("valid_mask.dtype={}",valid_mask.dtype)


    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[valid_mask],
    )

    tester.test(workload)
    

def test_sanity2():

    class s2(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self,boxes_tensor,scores_per_img ):
            
            valid_mask = torch.isfinite(boxes_tensor).all(dim=1) & torch.isfinite(
            scores_per_img
            )

            if not valid_mask.all():
                return valid_mask
            else :
                return valid_mask
                 
        
    model = s2()
    
    boxes_tensor = torch.load("boxes_tensor.pt",map_location="cpu")
    
    logger.info("boxes_tensor={}",boxes_tensor)
    logger.info("boxes_tensor.shape={}",boxes_tensor.shape)
    logger.info("boxes_tensor.dtype={}",boxes_tensor.dtype)
    
    scores_per_img = torch.load("scores_per_img.pt",map_location="cpu")

    logger.info("scores_per_img={}",scores_per_img)
    logger.info("scores_per_img.shape={}",scores_per_img.shape)
    logger.info("scores_per_img.dtype={}",scores_per_img.dtype)


    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[boxes_tensor,scores_per_img],
    )

    tester.test(workload)