# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger

def test_sdpa():

    class sdpa(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, query,key,value,attn_mask ):

            out = torch.nn.functional.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=	None,
                enable_gqa=	False,
                )
            
            return out


    model = sdpa().to(torch.bfloat16)
    model.eval()
    
    query = torch.randn(1, 28, 5224, 128,dtype=torch.bfloat16)
    key = torch.randn(1, 28, 5224, 128,dtype=torch.bfloat16)
    value = torch.randn(1, 28, 5224, 128,dtype=torch.bfloat16)
    attn_mask = torch.ones((1, 1, 1, 5224), dtype=torch.bool)
    attn_mask[..., -224:] = False  # padding -> must become -inf and be masked out


    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[query,key,value,attn_mask],
    )

    tester.test(workload)