# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
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
def test_qwen():
    class Qwen(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, grid_thw):
            
            cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
            )
            print("cu_seqlens", cu_seqlens)

            return cu_seqlens

    op = Qwen()

    full_key_states = torch.tensor([[ 1, 38, 58]])
    run_graph_test(
        op,
        [full_key_states],
        framework=Framework.TORCH,
    )