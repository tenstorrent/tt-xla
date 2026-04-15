# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger
import torch.nn.functional as F

def test_sdpa():

    class sdpa(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self,query, key, value  ):
            
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

            return hidden_states


    model = sdpa()
    
    query = torch.randn(2, 24, 4429, 64 ,dtype=torch.float32)
    key = torch.randn(2, 24, 4429, 64 ,dtype=torch.float32)
    value = torch.randn(2, 24, 4429, 64 ,dtype=torch.float32)


    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[query, key, value],
    )

    tester.test(workload)