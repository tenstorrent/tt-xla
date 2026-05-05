# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger
from torch import nn

def test_sanity():

    class sanity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.token_type_embeddings = nn.Embedding(2, 768)

        def forward(self, image_masks ):
            
            op =  self.token_type_embeddings(
            torch.full_like(image_masks, 1, dtype=torch.long)
            )
            return op


    model = sanity()

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)
    
    image_masks = torch.ones((1, 193), dtype=torch.int64)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[image_masks],
    )

    tester.test(workload)