# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity test: isolated cumsum op with exact inputs from the DeepseekOCR
masked scatter decomposition.

In the model forward, the cumsum operates on mask_i which is constructed as:
  mask = images_seq_mask[0].unsqueeze(-1)             # [913, 1] bool
  mask_broad, _ = broadcast_tensors(mask, embeds[0])  # [913, 1280] bool
  mask_flat = mask_broad.reshape(-1)                   # [1168640] bool
  mask_i = mask_flat.long()                            # [1168640] int64
  source_idx = torch.cumsum(mask_i, 0) - 1             # [1168640] int64

This test constructs that exact mask_i tensor from the real model inputs
and runs cumsum alone through run_op_test to confirm the op works in
isolation on the TT device.

"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test


class CumsumOp(nn.Module):
    """Wraps torch.cumsum exactly as used in masked scatter decomp."""

    def forward(self, mask_i):
        source_idx = torch.cumsum(mask_i, 0) - 1
        return source_idx


def _load_inputs():
    """Load model inputs and construct the exact mask_i for cumsum."""
    from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch import (
        ModelLoader,
    )

    loader = ModelLoader()
    full_model = loader.load_model(dtype_override=torch.bfloat16)
    raw_inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    images_seq_mask = raw_inputs["images_seq_mask"]  # [1, 913] bool
    input_ids = raw_inputs["input_ids"]  # [1, 913]

    embed_tokens = full_model.model.embed_tokens
    with torch.no_grad():
        inputs_embeds = embed_tokens(input_ids)  # [1, 913, 1280] bf16

    mask = images_seq_mask[0].unsqueeze(-1)  # [913, 1] bool
    mask_broad, _ = torch.broadcast_tensors(mask, inputs_embeds[0])  # [913, 1280]
    mask_flat = mask_broad.reshape(-1)  # [1168640] bool
    mask_i = mask_flat.long()  # [1168640] int64

    return mask_i


@pytest.fixture(scope="module")
def model_and_inputs():
    mask_i = _load_inputs()
    pipeline = CumsumOp()
    pipeline.eval()
    inputs = [mask_i]
    return pipeline, inputs


@pytest.mark.single_device
def test_deepseek_ocr_cumsum_alone(model_and_inputs):
    """
    Isolated cumsum with the exact mask_i from the DeepseekOCR
    masked scatter decomposition.

    Input:  mask_i [1168640] int64 (0s and 1s)
    Output: source_idx [1168640] int64

    If this passes → cumsum works fine alone, OOM is due to
    memory pressure from the larger compiled graph.
    If this OOMs → cumsum itself can't handle this input size
    on the TT device regardless of context.
    """
    pipeline, inputs = model_and_inputs

    run_op_test(
        pipeline,
        inputs,
        framework=Framework.TORCH,
    )
