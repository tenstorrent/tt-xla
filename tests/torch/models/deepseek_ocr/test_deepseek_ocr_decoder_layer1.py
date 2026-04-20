# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Isolated decoder layer 1 (MoE) sanity tests for DeepSeek OCR.

Layer 1 is the first MoE layer (DeepseekV2MoE, 64 routed experts, top-6).
The full layer hangs on TT — these tests break it into sub-blocks to
isolate which component causes the hang:

  1. Full decoder layer 1                (input_layernorm + attn + post_attn_layernorm + MoE)
  2. input_layernorm + attention only    (no MLP/MoE)
  3. post_attention_layernorm + MoE only (no attention)

Once the hanging sub-block is identified, it can be broken down further.
"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask as _prepare_4d_mask,
)


# ---------------------------------------------------------------------------
# Pipeline wrappers — each takes dummy hidden_states shaped like the real
# inputs_embeds [1, 913, 1280] and runs a piece of decoder layer 1.
# ---------------------------------------------------------------------------

class DecoderLayer1Full(nn.Module):
    """Full decoder layer 1 forward (layernorm + attn + layernorm + MoE)."""

    def __init__(self, ocr_model):
        super().__init__()
        self.layer = ocr_model.model.layers[1]

    def forward(self, hidden_states, attention_mask, position_ids):
        layer_outputs = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        return layer_outputs[0]


class DecoderLayer1AttnOnly(nn.Module):
    """Decoder layer 1: input_layernorm + self_attn + residual (skip MLP/MoE)."""

    def __init__(self, ocr_model):
        super().__init__()
        layer = ocr_model.model.layers[1]
        self.input_layernorm = layer.input_layernorm
        self.self_attn = layer.self_attn
        self.rotary_emb = layer.rotary_emb

    def forward(self, hidden_states, attention_mask, position_ids):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states
        return hidden_states


class DecoderLayer1MoEOnly(nn.Module):
    """Decoder layer 1: post_attention_layernorm + MoE MLP + residual (skip attn)."""

    def __init__(self, ocr_model):
        super().__init__()
        layer = ocr_model.model.layers[1]
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp

    def forward(self, hidden_states, attention_mask, position_ids):
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Model / input loading
# ---------------------------------------------------------------------------

def _load_model_and_inputs():
    import inspect
    import third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.loader as deepseek_ocr_loader
    from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch import ModelLoader
    from tests.runner.requirements import RequirementsManager

    loader_path = inspect.getsourcefile(deepseek_ocr_loader)
    with RequirementsManager.for_loader(loader_path):
        loader = ModelLoader()
        full_model = loader.load_model(dtype_override=torch.bfloat16)
        raw_inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    return full_model, raw_inputs


def _make_decoder_inputs(full_model, raw_inputs):
    """Build the (hidden_states, attention_mask, position_ids) that a
    decoder layer would receive — i.e. after embed_tokens + vision scatter."""
    input_ids = raw_inputs["input_ids"]
    hidden_states = full_model.model.embed_tokens(input_ids).to(torch.bfloat16)
    batch_size, seq_length = hidden_states.shape[:2]
    position_ids = torch.arange(
        0, seq_length, dtype=torch.long, device=hidden_states.device,
    ).unsqueeze(0)
    attention_mask = _prepare_4d_mask(
        None, (batch_size, seq_length), hidden_states, 0,
    )
    return [hidden_states, attention_mask, position_ids]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_model_and_raw_inputs():
    return _load_model_and_inputs()


@pytest.fixture(scope="module")
def decoder_inputs(full_model_and_raw_inputs):
    full_model, raw_inputs = full_model_and_raw_inputs
    return _make_decoder_inputs(full_model, raw_inputs)


@pytest.fixture(scope="module")
def layer1_full(full_model_and_raw_inputs, decoder_inputs):
    full_model, _ = full_model_and_raw_inputs
    pipeline = DecoderLayer1Full(full_model)
    pipeline.eval()
    pipeline = pipeline.to(torch.bfloat16)
    return pipeline, decoder_inputs


@pytest.fixture(scope="module")
def layer1_attn_only(full_model_and_raw_inputs, decoder_inputs):
    full_model, _ = full_model_and_raw_inputs
    pipeline = DecoderLayer1AttnOnly(full_model)
    pipeline.eval()
    pipeline = pipeline.to(torch.bfloat16)
    return pipeline, decoder_inputs


@pytest.fixture(scope="module")
def layer1_moe_only(full_model_and_raw_inputs, decoder_inputs):
    full_model, _ = full_model_and_raw_inputs
    pipeline = DecoderLayer1MoEOnly(full_model)
    pipeline.eval()
    pipeline = pipeline.to(torch.bfloat16)
    return pipeline, decoder_inputs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_decoder_layer1_full(layer1_full):
    """
    Full decoder layer 1 (input_layernorm + attn + post_attn_layernorm + MoE).
    Known to hang — baseline for sub-block isolation.
    """
    pipeline, inputs = layer1_full
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


@pytest.mark.single_device
def test_decoder_layer1_attn_only(layer1_attn_only):
    """
    Decoder layer 1: input_layernorm + self_attn + residual.
    Skips the MoE MLP entirely. If this passes, attention is fine and
    the MoE is the problem.
    """
    pipeline, inputs = layer1_attn_only
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


@pytest.mark.single_device
def test_decoder_layer1_moe_only(layer1_moe_only):
    """
    Decoder layer 1: post_attention_layernorm + MoE MLP + residual.
    Skips attention entirely. If this hangs, the DeepseekV2MoE block
    (64 routed experts, 2 shared experts, top-6 routing) is the root cause.
    """
    pipeline, inputs = layer1_moe_only
    run_op_test(pipeline, inputs, framework=Framework.TORCH)
