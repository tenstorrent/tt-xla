# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test, run_op_test_with_random_inputs
from utils import Category


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.topk",
)
@pytest.mark.parametrize(
    ["input_shape", "k"],
    [
        ((1, 10), 5),
        ((1, 20), 5),
        ((1, 30), 5),
        ((1, 40), 5),
        pytest.param(
            (1, 8400),
            300,
            marks=pytest.mark.xfail(
                reason="Bad PCC due to ttnn sort bug for greater than 256 elements - https://github.com/tenstorrent/tt-xla/issues/1797"
            ),
        ),
        pytest.param(
            (1, 50000),
            100,
            marks=pytest.mark.xfail(
                reason="Bad PCC due to ttnn sort bug for greater than 256 elements - https://github.com/tenstorrent/tt-xla/issues/1797"
            ),
        ),
    ],
)
def test_topk_indices(input_shape: tuple, k: int):
    """Test topk operation returning indices."""

    class TopKIndices(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, x):
            return torch.topk(x, self.k)[1]

    model = TopKIndices(k)

    run_op_test_with_random_inputs(
        model, [input_shape], dtype=torch.float32, framework=Framework.TORCH
    )


pytest.mark.nightly


@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.Tensor.topk",
)
def test_topk_rpn_proposals_real_tensors():
    """Test topk operation using explicitly exported true end-to-end model tensors."""

    class TopKAndSlice(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, logits, proposals):
            # 1. TopK extraction
            topk_scores, topk_idx = logits.topk(self.k, dim=1)

            # 2. Advanced indexing
            batch_size = logits.shape[0]
            batch_idx = torch.arange(batch_size, device=logits.device)
            topk_proposals = proposals[batch_idx[:, None], topk_idx]

            return topk_scores, topk_proposals

    # Using typical config from panoptic_fpn (pre_nms_topk=1000)
    model = TopKAndSlice(k=1000).to(torch.bfloat16)

    # Load explicit real model trace
    logits = torch.load("topk_logits_level_0.pt").to(torch.bfloat16)
    proposals = torch.load("topk_proposals_level_0.pt").to(torch.bfloat16)

    run_op_test(model, [logits, proposals], framework=Framework.TORCH)
