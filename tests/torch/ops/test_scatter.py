# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from utils import Category


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.scatter",
)
def test_scatter():

    class scatter(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, outputs_new, mask_leq, intermediate_tensor):
            outputs_new[mask_leq] = intermediate_tensor
            return outputs_new

    model = scatter()

    mask_leq = torch.tensor(
        [
            [
                True,
                True,
                False,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                False,
                False,
                True,
            ]
        ]
    )
    outputs_new = torch.tensor(
        [
            [
                0.50000,
                0.50000,
                0.50000,
                0.50000,
                0.50000,
                0.50000,
                0.50000,
                0.50000,
                0.50000,
                0.50000,
                0.50000,
                0.50000,
                0.50000,
                0.50000,
                0.50000,
                0.50000,
                0.50000,
                0.50000,
            ]
        ]
    )
    intermediate_tensor = torch.tensor(
        [
            0.32797,
            0.42933,
            0.28850,
            0.02414,
            0.27806,
            0.18570,
            0.24490,
            0.36455,
            0.00994,
            0.01115,
            0.27219,
        ]
    )

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)

    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[outputs_new, mask_leq, intermediate_tensor],
    )

    tester.test(workload)
