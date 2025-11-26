# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test_with_random_inputs
from utils import Category


@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.sum",
)
def test_sum_div():

    class sum_div(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.edge_weights = nn.Parameter(
                torch.tensor([1.2969, 0.5703], dtype=torch.bfloat16), requires_grad=True
            )

        def forward(self, nodes):

            weights_sum = torch.sum(self.edge_weights)
            out = (nodes * self.edge_weights[0]) / (weights_sum + 0.0001)
            return out

    model = sum_div().eval().to(torch.bfloat16)
    input_shape = (1, 64, 8, 8)

    run_op_test_with_random_inputs(
        model, [input_shape], dtype=torch.bfloat16, framework=Framework.TORCH
    )
