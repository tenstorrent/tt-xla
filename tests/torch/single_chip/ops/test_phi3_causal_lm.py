# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs, run_graph_test
from utils import Category
from torch import nn

@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_phi3_causal_lm():
    class Phi3_CausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = nn.Linear(32 * 96, 3072, bias=False)
            self.resid_attn_dropout = torch.nn.Dropout(p=0.0)
        def forward(self, attn_weights, value_states, residual):
            attn_weights= torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.bfloat16)
            attn_weights = nn.functional.dropout(attn_weights, p=0.0, training=False)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(1, 256, 3072)
            attn_output = self.o_proj(attn_output)
            hidden_states = residual + self.resid_attn_dropout(attn_output)
            print("attn_output:", attn_output)
            return hidden_states
           

    op = Phi3_CausalLM()

    op = op.to(torch.bfloat16)
    attn_weights=torch.load("attn_weights.pt")
    value_states=torch.load("value_states.pt")
    residual=torch.load("residual.pt")
    w = torch.load("/home/tt-xla/o_proj_weight.pt")
    with torch.no_grad():
        op.o_proj.weight.copy_(w)

    run_graph_test(
        op,
        [attn_weights, value_states, residual],
        framework=Framework.TORCH,
    )