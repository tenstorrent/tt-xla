# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity that reproduces the downstream `split_with_sizes` RuntimeError
hit by Qwen 3.5 27B (multimodal)

"""

import pytest
import torch
from infra import Framework, run_op_test
from utils import Category

SEQ_LEN = 1900
NUM_HEADS = 4
HEAD_DIM = 32


class SplitViaRepeatInterleave(torch.nn.Module):
    def forward(self, qkv, hw_prod, repeats):
        lengths = torch.repeat_interleave(hw_prod, repeats)
        splits = torch.split(qkv, lengths.tolist(), dim=2)
        return torch.stack([s.sum() for s in splits], dim=0)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_split_with_sizes_qwen35():
    qkv = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.bfloat16)
    hw_prod = torch.tensor([SEQ_LEN], dtype=torch.int64)   
    repeats = torch.tensor([1], dtype=torch.int64)         

    run_op_test(
        SplitViaRepeatInterleave(),
        [qkv, hw_prod, repeats],
        framework=Framework.TORCH,
    )
