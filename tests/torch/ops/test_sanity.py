# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
import numpy as np

def test_sanity():

    class reshape(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inputs ):
            index_dims = inputs.shape[1:-1]
            indices = np.prod(index_dims)
            output = torch.reshape(inputs, [1, indices, -1])
            return indices

    model = reshape()
    inputs = torch.randn(1, 224, 224, 256, dtype=torch.bfloat16)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[inputs],
    )

    tester.test(workload)
    
