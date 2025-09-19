# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from infra import run_graph_test_with_random_inputs, Framework
from utils import Category

# @pytest.mark.push
# @pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize("bias", [True])
def test_simple_matmul(bias):
    class MatMul(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 64, bias=bias)

        def forward(self, x):
            return self.linear(x)

    model = MatMul()
    run_graph_test_with_random_inputs(model, [(32, 32)], framework=Framework.TORCH)
