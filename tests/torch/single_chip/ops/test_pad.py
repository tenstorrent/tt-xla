# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from infra import Framework, run_op_test_with_random_inputs
from utils import Category
from loguru import logger


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.nn.functional.pad",
)
def test_pad():
    
    class PadModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x1):

            x1 = F.pad(x1, [-16,-16,-16,-16])
            
            return x1
    
    model = PadModule()
    model.to(torch.bfloat16)
    
    outputs = model(torch.randn(1, 1024, 64, 64,dtype=torch.bfloat16))
    
    logger.info("cpu inference done !")
    logger.info("outputs.shape ={}",outputs.shape)
    logger.info("outputs.dtype ={}",outputs.dtype)
    
    run_op_test_with_random_inputs(
        model, 
        [(1, 1024, 64, 64)], 
        dtype=torch.bfloat16, 
        framework=Framework.TORCH
    )
