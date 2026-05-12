# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test
from utils import Category


class MaskedScatter(torch.nn.Module):
    def forward(self, data, mask, source):
        return data.masked_scatter(mask, source)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_masked_scatter_row_constant():
    """Row-constant mask: every element along the last dim shares the same
    boolean value (the row is either fully True or fully False). This mirrors
    the multimodal image-token pattern used by Gemma3/LLaVA/DeepSeek-OCR."""
    seq_len, hidden = 32, 64
    n_true = 16

    data = torch.randn(1, seq_len, hidden, dtype=torch.bfloat16)
    source = torch.randn(1, n_true, hidden, dtype=torch.bfloat16)

    mask = torch.zeros(1, seq_len, hidden, dtype=torch.bool)
    mask[0, :n_true, :] = True

    run_op_test(
        MaskedScatter(),
        [data, mask, source],
        framework=Framework.TORCH,
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_masked_scatter_mixed_mask():
    """Mixed mask: True/False values are randomly distributed across elements
    (not row-constant), exercising the generic flat decomposition path."""
    data = torch.randn(4, 8, 16, dtype=torch.bfloat16)
    mask = torch.randint(0, 2, (4, 8, 16), dtype=torch.bool)
    source = torch.randn(int(mask.sum().item()), dtype=torch.bfloat16)

    run_op_test(
        MaskedScatter(),
        [data, mask, source],
        framework=Framework.TORCH,
    )
