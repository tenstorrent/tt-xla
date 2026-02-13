# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn.functional as F
from infra import Framework, run_graph_test
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_softmax_sanity():
    """Sanity test for F.softmax using saved attention scores"""

    class SoftmaxModule(torch.nn.Module):
        def forward(self, attention_scores):
            return F.softmax(attention_scores, dim=-1)

    # Load attention scores (input to softmax)
    attention_scores = torch.load("/home/tt-xla/attention_scores_cpu.pt")

    op = SoftmaxModule()

    run_graph_test(
        op,
        [attention_scores],
        framework=Framework.TORCH,
    )
