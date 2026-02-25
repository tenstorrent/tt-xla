# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, run_op_test
from infra.evaluators.evaluation_config import AtolConfig, PccConfig
from utils import Category


@pytest.mark.xfail(
    reason="TTNNWorkaroundsPass casts integer concat operands to bfloat16, "
    "rounding large token IDs (e.g. 19585 → 19584). "
    "Tracked: https://github.com/tenstorrent/tt-xla/issues/3463",
    strict=False,
)
@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_concat_int32():
    """torch.cat on int32 tensors must return exact values, not bfloat16-rounded.

    Regression test for a TT bug (TTNNWorkaroundsPass) where torch.cat on
    integer tensors in tile layout with non-32-aligned shapes inserts a
    cast to bfloat16 for padding, then casts back.  This rounds large integer
    values: e.g. 19585 → 19584 (bfloat16 precision is 64 units at that
    magnitude).  When two different top-k token IDs round to the same value
    their logprob dict entries collide and vLLM sees fewer logprob entries
    than requested.

    This test concatenates two int32 tensors containing values that are NOT
    exactly representable in bfloat16.  The output must exactly match the
    CPU reference.
    """

    class ConcatInt32(torch.nn.Module):
        def forward(self, a, b):
            return torch.cat([a, b], dim=1)

    # 19585 is not bfloat16-representable (rounds to 19584).
    # Use .float() on the output so it can be transferred from TT device
    # (TT cannot transfer standalone integer tensors from device).
    class ConcatInt32AsFloat(torch.nn.Module):
        def forward(self, a, b):
            return torch.cat([a, b], dim=1).float()

    a = torch.tensor([[264]], dtype=torch.int32)
    b = torch.tensor([[264, 280, 460, 19584, 19585]], dtype=torch.int32)

    # Use atol: PCC is undefined for near-constant outputs; atol catches any
    # rounding error (CPU=19585.0, TT=19584.0 → error=1.0 > atol=0.01).
    cfg = ComparisonConfig(atol=AtolConfig(True, 0.01), pcc=PccConfig(False))
    run_op_test(
        ConcatInt32AsFloat(), [a, b], framework=Framework.TORCH, comparison_config=cfg
    )
