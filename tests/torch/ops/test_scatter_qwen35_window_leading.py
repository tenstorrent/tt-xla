# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Focused sanity for the StableHLO -> TTIR scatter conversion when the
update_window_dim sits at a LEADING position (dim 0), not trailing.
"""

import pytest
import torch
from infra import Framework, run_op_test
from utils import Category


class IndexPutMRopePattern(torch.nn.Module):
    """Mirrors modeling_qwen3_5.py:1509:
        position_ids[:, 0, mask] = llm_positions
    where position_ids is (3, 1, N), mask selects N positions of the
    trailing dim, and llm_positions is (3, N).
    """

    def forward(self, position_ids, mask, llm_positions):
        out = position_ids.clone()
        out[:, 0, mask] = llm_positions
        return out


# Exact values from the failing model run: 3 = MRoPE axes (text/H/W),
# 494 = trailing seq_len after the prod-int bf16 fix.
MROPE_AXES = 3
SEQ_LEN_EXACT = 494


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_index_put_mrope_qwen35_exact():
    """Reproduction of the in-model failure with shapes matching the
    failing scatter op.

    Pass: device compiles and produces a result matching CPU.
    Fail before the fix:
        loc("scatter.N"): error: 'ttir.repeat' op Input tensor shape
            (494,1) at index 0 does not repeat to output (3,494) using
            repeat value 3.
    """
    position_ids = torch.zeros((MROPE_AXES, 1, SEQ_LEN_EXACT), dtype=torch.int64)
    mask = torch.ones(SEQ_LEN_EXACT, dtype=torch.bool)
    llm_positions = torch.arange(
        MROPE_AXES * SEQ_LEN_EXACT, dtype=torch.int64
    ).reshape(MROPE_AXES, SEQ_LEN_EXACT)

    run_op_test(
        IndexPutMRopePattern(),
        [position_ids, mask, llm_positions],
        framework=Framework.TORCH,
    )
