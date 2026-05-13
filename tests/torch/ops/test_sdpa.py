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

        def forward(self, query, key, value, attention_mask ):
            return torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)


    model = sdpa()
    query = torch.randn(1, 1, 1280, 1024, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 1280, 1024, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 1280, 1024, dtype=torch.bfloat16)
    attention_mask = torch.randn(1, 1280, 1280, dtype=torch.bfloat16)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[query, key, value, attention_mask],
    )

    tester.test(workload)
