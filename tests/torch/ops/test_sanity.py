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
            self.num_heads = 8
            self.head_dim = 32

        def forward(self, attn_probs, value_states):
            value_states = value_states.view(8, -1, 32)
            attn_output = torch.bmm(attn_probs, value_states)
            attn_output = attn_output.view(1, self.num_heads, 100, self.head_dim)
            return attn_output

    model = BmmWithViewPair()
    model.eval()

    attn_probs = torch.randn(8, 100, 100, dtype=torch.float32)
    value_states = torch.rand(1, 8, 100, 32, dtype=torch.float32)

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)
    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[attn_probs, value_states],
    )

    tester.test(workload)
