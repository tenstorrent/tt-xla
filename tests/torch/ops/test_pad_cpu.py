# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

def test_pad():

    class PadModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1):
            x1 = F.pad(x1, [-16,-16,-16,-16])
            return x1

    model = PadModule()
    model.to(torch.bfloat16)
    input = torch.randn(1, 1024, 64, 64, dtype=torch.bfloat16)
    logger.info(f"Input shape: {input.shape}")
    
    with torch.no_grad():
        output = model(input)
    
    logger.info(f"Output shape: {output.shape}")
