# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Expert dispatch bisection for DeepSeek OCR hang investigation.

test_moe_experts_only hangs — meaning the hang is in expert dispatch
(token sort + 64 expert MLPs + cat), not in unsort+weight.

This file bisects the 64-expert loop by running all experts but replacing
some with zeros (to keep output shape fixed for run_op_test comparison).

Tests:
  test_experts_1    — only expert 0 computes, rest produce zeros
  test_experts_8    — experts 0-7 compute, rest produce zeros
  test_experts_32   — experts 0-31 compute, rest produce zeros
  test_experts_all  — all 64 experts compute (= test_moe_experts_only)
"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask as _prepare_4d_mask,
)


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


def _vision_embed_forward(pipeline, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
    """Shared vision embedding logic."""
    inputs_embeds = pipeline.embed_tokens(input_ids)

    sam_model = pipeline.sam_model
    vision_model = pipeline.vision_model

    images = [(patches, image_ori)]

    if (
        sam_model is not None
        and (input_ids.shape[1] != 1 or pipeline.training)
        and torch.sum(images[0][1], dim=(0, 1, 2, 3)).item() != 0
    ):
        idx = 0
        for image, crop_shape in zip(images, images_spatial_crop):
            images_in_this_batch = []
            p = image[0]
            io = image[1]

            with torch.no_grad():
                if torch.sum(p).item() != 0:
                    lf1 = sam_model(p)
                    lf2 = vision_model(p, lf1)
                    local_features = torch.cat(
                        (lf2[:, 1:], lf1.flatten(2).permute(0, 2, 1)), dim=-1,
                    )
                    local_features = pipeline.projector(local_features)
                    gf1 = sam_model(io)
                    gf2 = vision_model(io, gf1)
                    global_features = torch.cat(
                        (gf2[:, 1:], gf1.flatten(2).permute(0, 2, 1)), dim=-1,
                    )
                    global_features = pipeline.projector(global_features)
                    _, hw, n_dim = global_features.shape
                    h = w = int(hw**0.5)
                    _2, hw2, n_dim2 = local_features.shape
                    h2 = w2 = int(hw2**0.5)
                    width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]

                    global_features = global_features.view(h, w, n_dim)
                    global_features = torch.cat(
                        [global_features, pipeline.image_newline[None, None, :].expand(h, 1, n_dim)],
                        dim=1,
                    )
                    global_features = global_features.view(-1, n_dim)

                    local_features = (
                        local_features.view(height_crop_num, width_crop_num, h2, w2, n_dim2)
                        .permute(0, 2, 1, 3, 4)
                        .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
                    )
                    local_features = torch.cat(
                        [local_features, pipeline.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2)],
                        dim=1,
                    )
                    local_features = local_features.view(-1, n_dim2)

                    global_local_features = torch.cat(
                        [local_features, global_features, pipeline.view_seperator[None, :]],
                        dim=0,
                    )
                else:
                    gf1 = sam_model(io)
                    gf2 = vision_model(io, gf1)
                    global_features = torch.cat(
                        (gf2[:, 1:], gf1.flatten(2).permute(0, 2, 1)), dim=-1,
                    )
                    global_features = pipeline.projector(global_features)
                    _, hw, n_dim = global_features.shape
                    h = w = int(hw**0.5)
                    global_features = global_features.view(h, w, n_dim)
                    global_features = torch.cat(
                        [global_features, pipeline.image_newline[None, None, :].expand(h, 1, n_dim)],
                        dim=1,
                    )
                    global_features = global_features.view(-1, n_dim)
                    global_local_features = torch.cat(
                        [global_features, pipeline.view_seperator[None, :]], dim=0
                    )

                images_in_this_batch.append(global_local_features)

            if images_in_this_batch:
                images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                inputs_embeds[idx] = _masked_scatter_decomp_muladd(
                    inputs_embeds[idx], images_seq_mask[idx], images_in_this_batch,
                )
            idx += 1

    return inputs_embeds


def _pre_decoder_setup(pipeline, inputs_embeds):
    """Compute position_ids and attention_mask."""
    batch_size, seq_length = inputs_embeds.shape[:2]
    position_ids = torch.arange(
        0, seq_length, dtype=torch.long, device=inputs_embeds.device,
    ).unsqueeze(0)

    if pipeline._use_flash_attention_2:
        attention_mask = None
    else:
        attention_mask = _prepare_4d_mask(
            None, (batch_size, seq_length), inputs_embeds, 0,
        )
    return attention_mask, position_ids


def _run_layer0(pipeline, hidden_states, attention_mask, position_ids):
    """Run layer 0 (dense MLP — known to pass)."""
    layer_outputs = pipeline.layer0(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    )
    return layer_outputs[0]


def _run_layer1_attn_half(pipeline, hidden_states, attention_mask, position_ids):
    """Run layer 1 attention half (input_layernorm + rotary + self_attn + residual)."""
    layer1 = pipeline.layer1
    residual = hidden_states
    hidden_states = layer1.input_layernorm(hidden_states)
    position_embeddings = layer1.rotary_emb(hidden_states, position_ids)
    hidden_states, _ = layer1.self_attn(
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


def _run_through_gate(pipeline, hidden_states):
    """Run post_attention_layernorm + gate, return tensors needed by moe_infer."""
    hidden_states = pipeline.layer1.post_attention_layernorm(hidden_states)
    moe = pipeline.layer1.mlp
    orig_shape = hidden_states.shape
    topk_idx, topk_weight, _ = moe.gate(hidden_states)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    return moe, hidden_states, topk_idx, topk_weight, orig_shape


# ---------------------------------------------------------------------------
# Modified moe_infer: run N experts, zeros for the rest
# ---------------------------------------------------------------------------

def _moe_infer_n_experts(moe, x, topk_ids, topk_weight, max_experts):
    """Expert loop where only the first max_experts compute; rest produce zeros.

    Output shape is always [total_dispatched_tokens, hidden_dim] regardless
    of max_experts, making it safe for run_op_test comparison.
    """
    cnts = topk_ids.new_zeros((topk_ids.shape[0], len(moe.experts)))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = x[idxs // topk_ids.shape[1]]

    tokens_per_expert = tokens_per_expert.cpu().tolist()

    outputs = []
    start_idx = 0
    experts_run = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        if experts_run < max_experts:
            expert = moe.experts[i + moe.ep_rank * moe.experts_per_rank]
            expert_out = expert(tokens_for_this_expert)
        else:
            expert_out = torch.zeros_like(tokens_for_this_expert)
        outputs.append(expert_out)
        start_idx = end_idx
        experts_run += 1

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    return outs


# ---------------------------------------------------------------------------
# Pipeline wrappers
# ---------------------------------------------------------------------------

class _BasePipeline(nn.Module):
    """Base class with shared components."""

    def __init__(self, ocr_model):
        super().__init__()
        self.embed_tokens = ocr_model.model.embed_tokens
        self.sam_model = ocr_model.model.sam_model
        self.vision_model = ocr_model.model.vision_model
        self.projector = ocr_model.model.projector
        self.image_newline = ocr_model.model.image_newline
        self.view_seperator = ocr_model.model.view_seperator
        self._use_flash_attention_2 = ocr_model.model._use_flash_attention_2
        self.layer0 = ocr_model.model.layers[0]
        self.layer1 = ocr_model.model.layers[1]


class _NExpertsPipeline(_BasePipeline):
    """Pipeline that runs only N experts (rest produce zeros)."""

    def __init__(self, ocr_model, max_experts):
        super().__init__(ocr_model)
        self.max_experts = max_experts

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        inputs_embeds = _vision_embed_forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop)
        attention_mask, position_ids = _pre_decoder_setup(self, inputs_embeds)
        hidden_states = _run_layer0(self, inputs_embeds, attention_mask, position_ids)
        hidden_states = _run_layer1_attn_half(self, hidden_states, attention_mask, position_ids)
        moe, hidden_states, topk_idx, topk_weight, orig_shape = _run_through_gate(self, hidden_states)
        outs = _moe_infer_n_experts(moe, hidden_states, topk_idx, topk_weight, self.max_experts)
        return (outs,)


# ---------------------------------------------------------------------------
# Helpers
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


def _extract_inputs(raw_inputs):
    return [
        raw_inputs["input_ids"],
        raw_inputs["images"][0][0],
        raw_inputs["images"][0][1],
        raw_inputs["images_seq_mask"],
        raw_inputs["images_spatial_crop"],
    ]


def _make_fixture(full_model_and_raw_inputs, max_experts):
    full_model, raw_inputs = full_model_and_raw_inputs
    pipeline = _NExpertsPipeline(full_model, max_experts)
    pipeline.eval()
    pipeline = pipeline.to(torch.bfloat16)
    return pipeline, _extract_inputs(raw_inputs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_model_and_raw_inputs():
    return _load_model_and_inputs()


@pytest.fixture(scope="module")
def model_0_experts(full_model_and_raw_inputs):
    return _make_fixture(full_model_and_raw_inputs, max_experts=0)


@pytest.fixture(scope="module")
def model_1_expert(full_model_and_raw_inputs):
    return _make_fixture(full_model_and_raw_inputs, max_experts=1)


@pytest.fixture(scope="module")
def model_8_experts(full_model_and_raw_inputs):
    return _make_fixture(full_model_and_raw_inputs, max_experts=8)


@pytest.fixture(scope="module")
def model_32_experts(full_model_and_raw_inputs):
    return _make_fixture(full_model_and_raw_inputs, max_experts=32)


@pytest.fixture(scope="module")
def model_64_experts(full_model_and_raw_inputs):
    return _make_fixture(full_model_and_raw_inputs, max_experts=64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_experts_0(model_0_experts):
    """
    Token sort + ALL zeros (no expert MLPs) + cat.
    Isolates token sort ops: scatter_, sum, argsort, fancy index, .cpu() graph break.
    If this hangs: token sort infrastructure is the problem.
    If this passes: the single expert MLP is the problem.
    """
    pipeline, inputs = model_0_experts
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


@pytest.mark.single_device
def test_experts_1(model_1_expert):
    """
    Token sort + 1 expert MLP + 63 zeros + cat.
    If this hangs: single expert MLP or token sort is the problem.
    """
    pipeline, inputs = model_1_expert
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


@pytest.mark.single_device
def test_experts_8(model_8_experts):
    """
    Token sort + 8 expert MLPs + 56 zeros + cat.
    If this hangs but test_experts_1 passes: hang scales with expert count.
    """
    pipeline, inputs = model_8_experts
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


@pytest.mark.single_device
def test_experts_32(model_32_experts):
    """
    Token sort + 32 expert MLPs + 32 zeros + cat.
    Halfway point for bisection.
    """
    pipeline, inputs = model_32_experts
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


@pytest.mark.single_device
def test_experts_all(model_64_experts):
    """
    Token sort + all 64 expert MLPs + cat.
    Same as test_moe_experts_only — expected to HANG.
    """
    pipeline, inputs = model_64_experts
    run_op_test(pipeline, inputs, framework=Framework.TORCH)
