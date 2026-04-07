# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT vs CPU PCC for the DeepSeek-OCR image-token scatter block (masked_scatter decomposition).

This duplicates the ``if images_in_this_batch:`` body in ``DeepseekOCRModel.forward`` after
``torch.cat`` — broadcast, flatten, ``cumsum`` gather, ``where`` — without importing
implementation from ``modeling_deepseekocr.py``. See https://github.com/tenstorrent/tt-xla/issues/3316
"""

import pytest
import torch
import torch.nn as nn
import torch_xla.runtime as xr
from infra import Framework, run_op_test
from utils import Category


class DeepseekOcrImageTokenScatterBlock(nn.Module):
    """Line-for-line copy of the scatter block (no shared helper in source)."""

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        images_seq_mask: torch.Tensor,
        images_in_this_batch: torch.Tensor,
    ) -> torch.Tensor:
        mask = images_seq_mask.unsqueeze(-1)
        mask_broad, data = torch.broadcast_tensors(mask, inputs_embeds)
        mask_flat = mask_broad.reshape(-1)
        data_flat = data.reshape(-1)
        source_flat = images_in_this_batch.reshape(-1)
        mask_i = mask_flat.long()
        source_idx = torch.cumsum(mask_i, 0) - 1
        source_idx = torch.clamp(source_idx, 0, source_flat.shape[0] - 1)
        gathered = source_flat[source_idx]
        result_flat = torch.where(mask_flat, gathered, data_flat)
        return result_flat.view_as(inputs_embeds)


def _make_inputs(
    seq_len: int,
    hidden: int,
    num_image_tokens: int,
    seed: int,
    dtype: torch.dtype,
):
    assert num_image_tokens <= seq_len
    g = torch.Generator().manual_seed(seed)
    inputs_embeds = torch.randn(seq_len, hidden, dtype=dtype, generator=g)
    mask = torch.zeros(seq_len, dtype=torch.bool)
    idx = torch.randperm(seq_len, generator=g)[:num_image_tokens]
    mask[idx] = True
    images_in_this_batch = torch.randn(num_image_tokens, hidden, dtype=dtype, generator=g)
    return inputs_embeds, mask, images_in_this_batch


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize(
    "seq_len,hidden,num_image_tokens",
    [
        (256, 1280, 128),
        (512, 1280, 200),
    ],
)
def test_deepseek_ocr_image_token_scatter_cpu_vs_tt_pcc(seq_len, hidden, num_image_tokens):
    """PCC between CPU and TT for the cumsum/gather/where scatter path."""
    xr.set_device_type("TT")
    dtype = torch.bfloat16
    inputs_embeds, mask, stacked = _make_inputs(
        seq_len, hidden, num_image_tokens, seed=42, dtype=dtype
    )

    run_op_test(
        DeepseekOcrImageTokenScatterBlock(),
        [inputs_embeds, mask, stacked],
        framework=Framework.TORCH
    )
