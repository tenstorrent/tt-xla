# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger
import numpy as np


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 224, 224, 256),
        # (1, 224, 224, 3) - redundant from model operation prespective 
        (1, 55, 55, 64),
        
    ],
)
def test_prod(input_shape):

    class prod(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inputs):
            index_dims = inputs.shape[1:-1]
            return np.prod(index_dims)

    model = prod()
    model.to(torch.bfloat16)
    model.eval()

    inputs = torch.randn(*input_shape, dtype=torch.bfloat16)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[inputs],
    )

    tester.test(workload)