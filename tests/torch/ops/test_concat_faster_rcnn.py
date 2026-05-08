# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger

def test_concat():

    class concat(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4 ):

            return torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)


    model = concat()
    
    pred_boxes1 = torch.randn(217413, 1, dtype=torch.float32)
    pred_boxes2 = torch.randn(217413, 1, dtype=torch.float32)
    pred_boxes3 = torch.randn(217413, 1, dtype=torch.float32)
    pred_boxes4 = torch.randn(217413, 1, dtype=torch.float32)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4],
    )

    tester.test(workload)