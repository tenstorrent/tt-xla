# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger

def test_index_put():

    class index_put(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self,  hidden_states,p0,p1, image_features_proj):
            
            hidden_states = hidden_states.index_put(
                (p0,p1), image_features_proj, accumulate=False
            )

            return hidden_states 


    model = index_put()
    hidden_states = torch.load('hidden_states.pt',map_location="cpu")
    
    p0= torch.load('positions_0.pt',map_location="cpu")
    p1 = torch.load('positions_1.pt',map_location="cpu")
    image_features_proj = torch.load('image_features_proj.pt',map_location="cpu")
    
    logger.info("hidden_states={}",hidden_states)
    logger.info("p0={}",p0)
    logger.info("p1={}",p1)
    logger.info("image_features_proj={}",image_features_proj)
    
    logger.info("hidden_states.shape={}",hidden_states.shape)
    logger.info("p0.shape={}",p0.shape)
    logger.info("p1.shape={}",p1.shape)
    logger.info("image_features_proj.shape={}",image_features_proj.shape)
    
    logger.info("hidden_states.dtype={}",hidden_states.dtype)
    logger.info("p0.dtype={}",p0.dtype)
    logger.info("p1.dtype={}",p1.dtype)
    logger.info("image_features_proj.dtype={}",image_features_proj.dtype)
    
    
    
    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[hidden_states,p0,p1,image_features_proj],
    )

    tester.test(workload)