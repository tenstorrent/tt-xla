# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs, run_graph_test
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_matmul_softmax():
    class Matmul(torch.nn.Module):
        def forward(self, x, y):
            attn_weights = torch.matmul(x,y.transpose(2, 3))*0.08838834764831845
            attn_weights= torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.bfloat16)
            return attn_weights

    op = Matmul()
    x=torch.load("key_states.pt",map_location="cpu")
    y=torch.load("query.pt",map_location="cpu")
    run_graph_test(
        op,
        [x,y],
        framework=Framework.TORCH,
    )