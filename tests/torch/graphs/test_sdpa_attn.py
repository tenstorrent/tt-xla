# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from infra import Framework, run_graph_test
from utils import Category
import torch.nn.functional as F

@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_scaled_dot_product_attention_basic():
    """Sanity test for torch.nn.functional.scaled_dot_product_attention"""

    class SDPAModule(torch.nn.Module):
        def __init__(self, dropout=0.0, scale=None, is_causal=False):
            super().__init__()
            self.dropout = dropout
            self.scale = scale
            self.is_causal = is_causal

        def forward(self, query, key, value, attn_mask):
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self.dropout,
                scale=self.scale,
                is_causal=self.is_causal,
            )

    query = torch.load("/home/tt-xla/query.pt")
    key = torch.load("/home/tt-xla/key.pt")
    value = torch.load("/home/tt-xla/value.pt")
    attn_mask = torch.load("/home/tt-xla/attention_mask.pt")

    op = SDPAModule(
        dropout=0.0,
        scale=0.11180339887498948,
        is_causal=False,
    )

    run_graph_test(
        op,
        [query, key, value, attn_mask],
        framework=Framework.TORCH,
    )
