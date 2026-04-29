# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Isolated moe_infer tests with varying upstream in XLA.

Confirmed: when upstream runs entirely on CPU and only moe_infer is in
the XLA graph, even 64 iterations (1 MLP + 63 zeros) do NOT hang.
The hang requires upstream ops in the pre-break XLA graph.

This file tests how much upstream is needed to trigger the hang by
progressively adding upstream layers into the XLA graph:

  Level 0 (no upstream)  -- moe_infer only          -- CONFIRMED: no hang
  Level 1 (layer1 only)  -- layer1_attn + gate + moe_infer
  Level 2 (layer0+1)     -- layer0 + layer1_attn + gate + moe_infer
  Level 3 (full)         -- vision + layer0 + layer1 -- CONFIRMED: hangs at 48

All use 48 iters (1 MLP + 47 zeros) since that's the known hang threshold
with full upstream.

Tests:
  test_layer1_only_48    -- Level 1: layer1_attn + gate + moe (48 iters)
  test_layer0_plus_1_48  -- Level 2: layer0 + layer1_attn + gate + moe (48 iters)
"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask as _prepare_4d_mask,
)


# ---------------------------------------------------------------------------
# Shared: masked_scatter decomp (same as in test_moe_infer_sanity.py)
# ---------------------------------------------------------------------------

def _masked_scatter_decomp_muladd(inputs_embeds_row, mask_1d, source):
    S, D = inputs_embeds_row.shape
    mask_i = mask_1d.long()
    source_idx = torch.cumsum(mask_i, 0) - 1
    source_idx = torch.clamp(source_idx, 0, source.shape[0] - 1)
    flat_source = source.reshape(-1)
    col_idx = torch.arange(D, device=source.device, dtype=source_idx.dtype)
    flat_idx = source_idx.unsqueeze(-1) * D + col_idx.unsqueeze(0)
    gathered_rows = flat_source[flat_idx.reshape(-1)].reshape(S, D)
    return torch.where(mask_1d.unsqueeze(-1), gathered_rows, inputs_embeds_row)


# ---------------------------------------------------------------------------
# Shared: moe_infer expert loop (1 MLP + N-1 zeros)
# ---------------------------------------------------------------------------

def _moe_infer_mlp_plus_zeros(moe, x, topk_idx, max_iters):
    cnts = topk_idx.new_zeros((topk_idx.shape[0], len(moe.experts)))
    cnts.scatter_(1, topk_idx, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_idx.view(-1).argsort()
    sorted_tokens = x[idxs // topk_idx.shape[1]]

    tokens_per_expert = tokens_per_expert.cpu().tolist()

    outputs = []
    start_idx = 0
    iters_done = 0
    mlp_done = False
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        if iters_done >= max_iters:
            break
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        if not mlp_done:
            expert = moe.experts[i + moe.ep_rank * moe.experts_per_rank]
            expert_out = expert(tokens_for_this_expert)
            mlp_done = True
        else:
            expert_out = torch.zeros_like(tokens_for_this_expert)
        outputs.append(expert_out)
        start_idx = end_idx
        iters_done += 1

    return torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)


# ---------------------------------------------------------------------------
# Level 1: Only layer1 (attn + gate + moe) in XLA
# Input: hidden_states after layer0 (computed on CPU)
# ---------------------------------------------------------------------------

class Layer1OnlyPipeline(nn.Module):
    """layer1_attn + layernorm + gate + moe_infer, all in XLA.
    attention_mask and position_ids are pre-computed on CPU and passed as inputs
    to avoid _prepare_4d_mask triggering compiler issues inside the XLA graph.
    """

    def __init__(self, ocr_model, max_iters):
        super().__init__()
        self.layer1 = ocr_model.model.layers[1]
        self.max_iters = max_iters

    def forward(self, hidden_states, attention_mask, position_ids):
        layer1 = self.layer1
        residual = hidden_states
        hs = layer1.input_layernorm(hidden_states)
        pos_emb = layer1.rotary_emb(hs, position_ids)
        hs, _ = layer1.self_attn(
            hidden_states=hs, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=None,
            output_attentions=False, use_cache=False,
            position_embeddings=pos_emb,
        )
        hidden_states = residual + hs

        hidden_states = layer1.post_attention_layernorm(hidden_states)
        moe = layer1.mlp
        topk_idx, topk_weight, _ = moe.gate(hidden_states)
        x = hidden_states.view(-1, hidden_states.shape[-1])

        return _moe_infer_mlp_plus_zeros(moe, x, topk_idx, self.max_iters)


# ---------------------------------------------------------------------------
# Level 2: layer0 + layer1 in XLA
# Input: inputs_embeds after vision_embed (computed on CPU)
# ---------------------------------------------------------------------------

class Layer0Plus1Pipeline(nn.Module):
    """layer0 + layer1_attn + layernorm + gate + moe_infer, all in XLA.
    attention_mask and position_ids are pre-computed on CPU and passed as inputs.
    """

    def __init__(self, ocr_model, max_iters):
        super().__init__()
        self.layer0 = ocr_model.model.layers[0]
        self.layer1 = ocr_model.model.layers[1]
        self.max_iters = max_iters

    def forward(self, inputs_embeds, attention_mask, position_ids):
        hidden_states = inputs_embeds
        layer_out = self.layer0(
            hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=None, output_attentions=False, use_cache=False,
        )
        hidden_states = layer_out[0]

        layer1 = self.layer1
        residual = hidden_states
        hs = layer1.input_layernorm(hidden_states)
        pos_emb = layer1.rotary_emb(hs, position_ids)
        hs, _ = layer1.self_attn(
            hidden_states=hs, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=None,
            output_attentions=False, use_cache=False,
            position_embeddings=pos_emb,
        )
        hidden_states = residual + hs

        hidden_states = layer1.post_attention_layernorm(hidden_states)
        moe = layer1.mlp
        topk_idx, topk_weight, _ = moe.gate(hidden_states)
        x = hidden_states.view(-1, hidden_states.shape[-1])

        return _moe_infer_mlp_plus_zeros(moe, x, topk_idx, self.max_iters)


# ---------------------------------------------------------------------------
# CPU upstream helpers
# ---------------------------------------------------------------------------

def _load_model_and_inputs():
    import inspect
    import third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.loader as loader_mod
    from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch import ModelLoader
    from tests.runner.requirements import RequirementsManager

    loader_path = inspect.getsourcefile(loader_mod)
    with RequirementsManager.for_loader(loader_path):
        loader = ModelLoader()
        full_model = loader.load_model(dtype_override=torch.bfloat16)
        raw_inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    return full_model, raw_inputs


def _run_vision_embed_on_cpu(model, raw_inputs):
    """Run vision_embed on CPU, return inputs_embeds [1, seq_len, hidden_dim]."""
    m = model.model
    input_ids = raw_inputs["input_ids"]
    images = raw_inputs["images"]
    images_seq_mask = raw_inputs["images_seq_mask"]
    images_spatial_crop = raw_inputs["images_spatial_crop"]

    with torch.no_grad():
        inputs_embeds = m.embed_tokens(input_ids)
        patches = images[0][0]
        image_ori = images[0][1]

        if (
            m.sam_model is not None
            and input_ids.shape[1] != 1
            and torch.sum(image_ori, dim=(0, 1, 2, 3)).item() != 0
        ):
            for idx, (image, crop_shape) in enumerate(
                zip([(patches, image_ori)], images_spatial_crop)
            ):
                p, io = image[0], image[1]
                if torch.sum(p).item() != 0:
                    lf1 = m.sam_model(p)
                    lf2 = m.vision_model(p, lf1)
                    local_features = torch.cat(
                        (lf2[:, 1:], lf1.flatten(2).permute(0, 2, 1)), dim=-1,
                    )
                    local_features = m.projector(local_features)
                    gf1 = m.sam_model(io)
                    gf2 = m.vision_model(io, gf1)
                    global_features = torch.cat(
                        (gf2[:, 1:], gf1.flatten(2).permute(0, 2, 1)), dim=-1,
                    )
                    global_features = m.projector(global_features)
                    _, hw, n_dim = global_features.shape
                    h = w = int(hw ** 0.5)
                    _2, hw2, n_dim2 = local_features.shape
                    h2 = w2 = int(hw2 ** 0.5)
                    wcn, hcn = crop_shape[0], crop_shape[1]

                    gf = global_features.view(h, w, n_dim)
                    gf = torch.cat(
                        [gf, m.image_newline[None, None, :].expand(h, 1, n_dim)],
                        dim=1,
                    ).view(-1, n_dim)

                    lf = (
                        local_features.view(hcn, wcn, h2, w2, n_dim2)
                        .permute(0, 2, 1, 3, 4)
                        .reshape(hcn * h2, wcn * w2, n_dim2)
                    )
                    lf = torch.cat(
                        [lf, m.image_newline[None, None, :].expand(hcn * h2, 1, n_dim2)],
                        dim=1,
                    ).view(-1, n_dim2)

                    glf = torch.cat([lf, gf, m.view_seperator[None, :]], dim=0)

                    inputs_embeds[idx] = _masked_scatter_decomp_muladd(
                        inputs_embeds[idx], images_seq_mask[idx], glf,
                    )

    return inputs_embeds.detach()


def _compute_mask_and_position_ids(model, inputs_embeds):
    """Pre-compute attention_mask and position_ids on CPU."""
    m = model.model
    batch_size, seq_length = inputs_embeds.shape[:2]
    position_ids = torch.arange(0, seq_length, dtype=torch.long).unsqueeze(0)

    if m._use_flash_attention_2:
        attention_mask = None
    else:
        attention_mask = _prepare_4d_mask(
            None, (batch_size, seq_length), inputs_embeds, 0,
        )

    return attention_mask, position_ids


def _run_through_layer0_on_cpu(model, inputs_embeds, attention_mask, position_ids):
    """Run layer0 on CPU, return hidden_states after layer0."""
    m = model.model
    with torch.no_grad():
        layer_out = m.layers[0](
            inputs_embeds, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=None, output_attentions=False, use_cache=False,
        )
    return layer_out[0].detach()


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_and_intermediates():
    """Load model, run vision_embed and layer0 on CPU.
    Returns (full_model, inputs_embeds, hidden_after_layer0, attention_mask, position_ids).
    """
    full_model, raw_inputs = _load_model_and_inputs()
    inputs_embeds = _run_vision_embed_on_cpu(full_model, raw_inputs)
    attention_mask, position_ids = _compute_mask_and_position_ids(full_model, inputs_embeds)
    hidden_after_layer0 = _run_through_layer0_on_cpu(
        full_model, inputs_embeds, attention_mask, position_ids,
    )
    return full_model, inputs_embeds, hidden_after_layer0, attention_mask, position_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_layer1_only_48(model_and_intermediates):
    """Level 1: only layer1 in XLA (vision + layer0 on CPU). 48 iters."""
    full_model, _, hidden_after_layer0, attention_mask, position_ids = model_and_intermediates
    pipeline = Layer1OnlyPipeline(full_model, max_iters=48).eval().to(torch.bfloat16)
    run_op_test(
        pipeline,
        [hidden_after_layer0, attention_mask, position_ids],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
def test_layer0_plus_1_48(model_and_intermediates):
    """Level 2: layer0 + layer1 in XLA (vision on CPU). 48 iters."""
    full_model, inputs_embeds, _, attention_mask, position_ids = model_and_intermediates
    pipeline = Layer0Plus1Pipeline(full_model, max_iters=48).eval().to(torch.bfloat16)
    run_op_test(
        pipeline,
        [inputs_embeds, attention_mask, position_ids],
        framework=Framework.TORCH,
    )
