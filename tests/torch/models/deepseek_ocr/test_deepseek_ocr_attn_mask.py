# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Isolated sanity for _prepare_4d_causal_attention_mask with the exact same
inputs and shapes as the DeepseekOCR whole-model forward.

In the whole model, OOM appears when this attention mask is added to the
pipeline (test_deepseek_ocr_vision_embed_preloop). This sanity runs
_prepare_4d_causal_attention_mask alone to verify it works on TT device.

Call site in DeepseekV2Model.forward (line 678):
    attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask,              # None
        (batch_size, seq_length),    # (1, 913)
        inputs_embeds,               # [1, 913, 1280] bf16
        past_key_values_length,      # 0
    )

Since attention_mask=None, this hits the else branch:
    AttentionMaskConverter(is_causal=True).to_causal_4d(
        batch_size=1, query_length=913, key_value_length=913,
        dtype=bf16, device=inputs_embeds.device
    )
Which calls _make_causal_mask((1,913), bf16, device, 0):
    - torch.full((913,913), finfo(bf16).min)
    - masked_fill_ lower triangle to 0
    - expand to [1,1,913,913]

Output: [1, 1, 913, 913] bf16

"""

import pytest
import torch
import torch.nn as nn
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from infra import Framework, run_op_test

BATCH_SIZE = 1
SEQ_LENGTH = 913
HIDDEN_SIZE = 1280
DTYPE = torch.bfloat16


class Prepare4dCausalAttentionMask(nn.Module):
    """Wraps _prepare_4d_causal_attention_mask as an nn.Module.

    Takes inputs_embeds as the only input (attention_mask is None,
    past_key_values_length is 0 — matching the whole model).
    """

    def forward(self, inputs_embeds):
        batch_size, seq_length = inputs_embeds.shape[:2]

        attention_mask = _prepare_4d_causal_attention_mask(
            None,
            (batch_size, seq_length),
            inputs_embeds,
            0,
        )

        return attention_mask


@pytest.fixture
def model_inputs():
    torch.manual_seed(42)
    inputs_embeds = torch.randn(
        BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE, dtype=DTYPE
    )
    return [inputs_embeds]


# ---------------------------------------------------------------------------
# Sanity: _prepare_4d_causal_attention_mask alone (CPU vs TT)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_prepare_4d_causal_attention_mask(model_inputs):
    """
    _prepare_4d_causal_attention_mask with the exact same shapes and params
    as the DeepseekOCR whole-model forward.

    Input:  inputs_embeds [1, 913, 1280] bf16
    Params: attention_mask=None, input_shape=(1, 913), past_kv_len=0
    Output: [1, 1, 913, 913] bf16 causal mask
    """
    model = Prepare4dCausalAttentionMask()
    model.eval()

    run_op_test(
        model,
        model_inputs,
        framework=Framework.TORCH,
    )
