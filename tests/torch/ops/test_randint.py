# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger

def test_randint():

    class randint(torch.nn.Module):
        def forward(self, input_ids):
            decoder_input_ids = torch.randint(
                low=0,
                high=262144,
                size=(input_ids.shape[0],256),
                device=input_ids.device,  
            )

            return decoder_input_ids


    model = randint()

    input_ids = torch.tensor(
    [[
        2, 105, 2364, 107, 11355, 563, 506, 7217, 3730,
        236881, 106, 107, 105, 4368, 107, 100, 45518, 107, 101
    ]],
    dtype=torch.int64,
    )

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[input_ids],
    )

    tester.test(workload)
