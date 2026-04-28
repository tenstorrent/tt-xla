# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Layer 1 bisection tests for DeepSeek OCR hang investigation.

Uses the same full-pipeline approach as the working layer tests
(vision_embed + setup + layer0 + layer1 sub-modules all in one forward),
but stops at different points within layer 1 to isolate the hang.

DeepseekV2DecoderLayer forward order:
  1. input_layernorm (RMSNorm)
  2. rotary_emb → position embeddings
  3. self_attn (LlamaAttention) + residual
  4. post_attention_layernorm (RMSNorm)
  5. mlp = DeepseekV2MoE:
       a. gate (MoEGate — routing, topk)
       b. moe_infer (token sort → 64 routed experts → unsort + weight)
       c. shared_experts (single DeepseekV2MLP)
  6. residual

Tests (each includes full vision_embed + setup + layer0 in the graph):
  test_layer1_attn_only       — layer0 + layer1 attention half (no MoE)
  test_layer1_moe_gate        — layer0 + layer1 full + gate routing only
  test_layer1_shared_experts  — layer0 + layer1 attn + shared experts only
  test_layer1_moe_infer       — layer0 + layer1 attn + gate + routed experts
  test_layer1_moe_block       — layer0 + layer1 attn + full MoE (gate+routed+shared)
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
    """Shared vision embedding logic (same as DeepseekOCRVisionEmbedPipeline)."""
    attention_mask = None
    position_ids = None

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
    """Compute position_ids and attention_mask (same as DeepseekV2Model.forward setup)."""
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
    """Run layer 1 attention half only (input_layernorm + rotary + self_attn + residual)."""
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


# ---------------------------------------------------------------------------
# Pipeline wrappers — each includes full vision_embed + setup + layer0
# ---------------------------------------------------------------------------

class _BaseBisectPipeline(nn.Module):
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


class Layer1AttnOnlyPipeline(_BaseBisectPipeline):
    """Step 1: layer0 + layer1 attn only.

    Runs: input_layernorm → rotary_emb → self_attn → residual add
    Matches lines 500-536 of modeling_deepseekv2.py
    """

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        inputs_embeds = _vision_embed_forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop)
        attention_mask, position_ids = _pre_decoder_setup(self, inputs_embeds)
        hidden_states = _run_layer0(self, inputs_embeds, attention_mask, position_ids)
        hidden_states = _run_layer1_attn_half(self, hidden_states, attention_mask, position_ids)
        return (hidden_states,)


class Layer1PostAttnLayerNormPipeline(_BaseBisectPipeline):
    """Step 2: layer0 + layer1 attn + post_attention_layernorm.

    Adds: post_attention_layernorm (RMSNorm)
    Matches line 539 of modeling_deepseekv2.py
    """

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        inputs_embeds = _vision_embed_forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop)
        attention_mask, position_ids = _pre_decoder_setup(self, inputs_embeds)
        hidden_states = _run_layer0(self, inputs_embeds, attention_mask, position_ids)
        hidden_states = _run_layer1_attn_half(self, hidden_states, attention_mask, position_ids)
        hidden_states = self.layer1.post_attention_layernorm(hidden_states)
        return (hidden_states,)


class Layer1GatePipeline(_BaseBisectPipeline):
    """Step 3: layer0 + layer1 attn + post_attention_layernorm + gate.

    Adds: MoEGate (linear → softmax/sigmoid → topk routing)
    Matches line 340 of modeling_deepseekv2.py (MoE.forward)
    """

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        inputs_embeds = _vision_embed_forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop)
        attention_mask, position_ids = _pre_decoder_setup(self, inputs_embeds)
        hidden_states = _run_layer0(self, inputs_embeds, attention_mask, position_ids)
        hidden_states = _run_layer1_attn_half(self, hidden_states, attention_mask, position_ids)
        hidden_states = self.layer1.post_attention_layernorm(hidden_states)
        topk_idx, topk_weight, _ = self.layer1.mlp.gate(hidden_states)
        return topk_weight


class Layer1MoeInferPipeline(_BaseBisectPipeline):
    """Step 4: layer0 + layer1 attn + post_attention_layernorm + gate + moe_infer.

    Adds: moe_infer (scatter → argsort → 64 routed expert MLPs → unsort → weight)
    Matches lines 340-354 of modeling_deepseekv2.py (MoE.forward)
    """

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        inputs_embeds = _vision_embed_forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop)
        attention_mask, position_ids = _pre_decoder_setup(self, inputs_embeds)
        hidden_states = _run_layer0(self, inputs_embeds, attention_mask, position_ids)
        hidden_states = _run_layer1_attn_half(self, hidden_states, attention_mask, position_ids)
        hidden_states = self.layer1.post_attention_layernorm(hidden_states)

        moe = self.layer1.mlp
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, _ = moe.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = moe.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        return (y,)


class Layer1FullMoEPipeline(_BaseBisectPipeline):
    """Step 5: layer0 + layer1 attn + post_attention_layernorm + gate + moe_infer + shared_experts + residual.

    Adds: shared_experts(identity) + residual connection
    Matches lines 337-357 + 538-541 of modeling_deepseekv2.py
    This is the complete layer1 = full DeepseekV2DecoderLayer.forward
    """

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        inputs_embeds = _vision_embed_forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop)
        attention_mask, position_ids = _pre_decoder_setup(self, inputs_embeds)
        hidden_states = _run_layer0(self, inputs_embeds, attention_mask, position_ids)
        hidden_states = _run_layer1_attn_half(self, hidden_states, attention_mask, position_ids)

        residual = hidden_states
        hidden_states = self.layer1.post_attention_layernorm(hidden_states)
        hidden_states = self.layer1.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states,)


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


def _make_bisect_fixture(full_model_and_raw_inputs, pipeline_cls):
    full_model, raw_inputs = full_model_and_raw_inputs
    pipeline = pipeline_cls(full_model)
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
def step1_model(full_model_and_raw_inputs):
    return _make_bisect_fixture(full_model_and_raw_inputs, Layer1AttnOnlyPipeline)


@pytest.fixture(scope="module")
def step2_model(full_model_and_raw_inputs):
    return _make_bisect_fixture(full_model_and_raw_inputs, Layer1PostAttnLayerNormPipeline)


@pytest.fixture(scope="module")
def step3_model(full_model_and_raw_inputs):
    return _make_bisect_fixture(full_model_and_raw_inputs, Layer1GatePipeline)


@pytest.fixture(scope="module")
def step4_model(full_model_and_raw_inputs):
    return _make_bisect_fixture(full_model_and_raw_inputs, Layer1MoeInferPipeline)


@pytest.fixture(scope="module")
def step5_model(full_model_and_raw_inputs):
    return _make_bisect_fixture(full_model_and_raw_inputs, Layer1FullMoEPipeline)


# ---------------------------------------------------------------------------
# Step 1: attn only
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_step1_attn_only(step1_model):
    """
    layer0 + layer1: input_layernorm → rotary_emb → self_attn → residual
    Same attention as layer 0 — expected to PASS.
    """
    pipeline, inputs = step1_model
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Step 2: attn + post_attention_layernorm
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_step2_post_attn_layernorm(step2_model):
    """
    layer0 + layer1 attn + post_attention_layernorm (RMSNorm).
    Adds one RMSNorm — should PASS.
    """
    pipeline, inputs = step2_model
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Step 3: attn + post_attention_layernorm + gate
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_step3_gate(step3_model):
    """
    layer0 + layer1 attn + post_attention_layernorm + MoEGate.
    Gate = linear → softmax/sigmoid → topk routing.
    """
    pipeline, inputs = step3_model
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Step 4: attn + post_attention_layernorm + gate + moe_infer
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_step4_moe_infer(step4_model):
    """
    layer0 + layer1 attn + post_attention_layernorm + gate + moe_infer.
    moe_infer = scatter → argsort → 64 routed expert MLPs → unsort → weight.
    Most likely hang culprit.
    """
    pipeline, inputs = step4_model
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Step 5: attn + post_attention_layernorm + gate + moe_infer + shared_experts + residual
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_step5_full_layer1(step5_model):
    """
    layer0 + full layer1 = attn + post_layernorm + gate + moe_infer + shared_experts + residual.
    Complete DeepseekV2DecoderLayer.forward for layer 1.
    Expected to HANG.
    """
    pipeline, inputs = step5_model
    run_op_test(pipeline, inputs, framework=Framework.TORCH)
