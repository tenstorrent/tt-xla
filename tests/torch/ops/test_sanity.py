# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester


def test_bmm_view_rank_mismatch():

    class BmmWithViewPair(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 12
            self.head_dim = 64

        def forward(self, attn_probs, value_states):
            value_states = value_states.view(12, -1, 64)
            attn_output = torch.bmm(attn_probs, value_states)
            attn_output = attn_output.view(1, self.num_heads, 577, self.head_dim)
            return attn_output

    model = BmmWithViewPair()
    model.eval()

    attn_probs = torch.randn(12, 577, 577, dtype=torch.bfloat16)
    value_states = torch.randn(1, 12, 577, 64, dtype=torch.bfloat16)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)
    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[attn_probs, value_states],
    )

    tester.test(workload)
