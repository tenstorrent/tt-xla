# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
moe_infer bisection tests for DeepSeek OCR hang investigation.

test_step4_moe_infer hangs. This file bisects inside moe_infer to find
the exact op/block that causes the hang.

moe_infer flow (ep_size=1, from modeling_deepseekv2.py lines 360-433):

  Stage A — Token routing prep (lines 361-366):
    cnts = topk_ids.new_zeros(...)             # [num_tokens, 64]
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)        # [64]
    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = x[idxs // topk_ids.shape[1]]

  Stage B — Expert dispatch loop (lines 401-411):
    for i, num_tokens in enumerate(tokens_per_expert):
        tokens_for_this_expert = sorted_tokens[start:end]  # dynamic slice
        expert_out = expert(tokens_for_this_expert)         # DeepseekV2MLP

  Stage C — Output reassembly (lines 413-425):
    outs = torch.cat(outputs, dim=0)
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs                        # unsort

  Stage D — Weighted sum (lines 426-432):
    final = new_x.view(...).type(...).mul_(topk_weight...).sum(1).type(...)

Tests:
  test_moe_stage_a          — token routing prep only
  test_moe_stage_ab_1expert — + 1 expert MLP (expert 0)
  test_moe_stage_ab_all     — + all 64 expert MLPs (full loop)
  test_moe_stage_abc        — + output reassembly (cat + unsort)
  test_moe_stage_abcd       — + weighted sum = full moe_infer
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
            p, io = image[0], image[1]
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
                        [global_features, pipeline.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1,
                    )
                    global_features = global_features.view(-1, n_dim)
                    local_features = (
                        local_features.view(height_crop_num, width_crop_num, h2, w2, n_dim2)
                        .permute(0, 2, 1, 3, 4)
                        .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
                    )
                    local_features = torch.cat(
                        [local_features, pipeline.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2)], dim=1,
                    )
                    local_features = local_features.view(-1, n_dim2)
                    global_local_features = torch.cat(
                        [local_features, global_features, pipeline.view_seperator[None, :]], dim=0,
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
                        [global_features, pipeline.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1,
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


def _run_through_layer0_and_layer1_attn_and_gate(pipeline, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
    """Run full pipeline up to and including layer1 post_attention_layernorm + gate.

    Returns (hidden_states_flat, topk_idx, topk_weight, orig_shape) — the exact
    inputs to moe_infer as they appear in DeepseekV2MoE.forward lines 340-341.
    """
    inputs_embeds = _vision_embed_forward(pipeline, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop)

    batch_size, seq_length = inputs_embeds.shape[:2]
    position_ids = torch.arange(0, seq_length, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)
    if pipeline._use_flash_attention_2:
        attention_mask = None
    else:
        attention_mask = _prepare_4d_mask(None, (batch_size, seq_length), inputs_embeds, 0)

    # layer 0
    hidden_states = inputs_embeds
    layer_outputs = pipeline.layer0(
        hidden_states, attention_mask=attention_mask, position_ids=position_ids,
        past_key_value=None, output_attentions=False, use_cache=False,
    )
    hidden_states = layer_outputs[0]

    # layer 1 attention half
    layer1 = pipeline.layer1
    residual = hidden_states
    hs = layer1.input_layernorm(hidden_states)
    position_embeddings = layer1.rotary_emb(hs, position_ids)
    hs, _ = layer1.self_attn(
        hidden_states=hs, attention_mask=attention_mask, position_ids=position_ids,
        past_key_value=None, output_attentions=False, use_cache=False,
        position_embeddings=position_embeddings,
    )
    hidden_states = residual + hs

    # post_attention_layernorm + gate
    hidden_states = layer1.post_attention_layernorm(hidden_states)
    moe = layer1.mlp
    orig_shape = hidden_states.shape
    topk_idx, topk_weight, _ = moe.gate(hidden_states)
    hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])

    return hidden_states_flat, topk_idx, topk_weight, orig_shape


# ---------------------------------------------------------------------------
# Pipeline wrappers — each inlines progressively more of moe_infer
# ---------------------------------------------------------------------------

class _BaseMoeInferPipeline(nn.Module):
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


class MoeStageAPipeline(_BaseMoeInferPipeline):
    """Stage A: Token routing prep only.

    cnts = new_zeros → scatter_ → sum (tokens_per_expert)
    idxs = argsort
    sorted_tokens = x[idxs // topk_ids.shape[1]]
    """

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        x, topk_idx, topk_weight, orig_shape = _run_through_layer0_and_layer1_attn_and_gate(
            self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop,
        )
        moe = self.layer1.mlp

        cnts = topk_idx.new_zeros((topk_idx.shape[0], len(moe.experts)))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = x[idxs // topk_idx.shape[1]]

        return sorted_tokens


class MoeStageAB1ExpertPipeline(_BaseMoeInferPipeline):
    """Stage A + B (1 expert): Token routing + single expert MLP.

    Runs only expert 0 on its token slice to test if a single expert hangs.
    """

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        x, topk_idx, topk_weight, orig_shape = _run_through_layer0_and_layer1_attn_and_gate(
            self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop,
        )
        moe = self.layer1.mlp

        cnts = topk_idx.new_zeros((topk_idx.shape[0], len(moe.experts)))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = x[idxs // topk_idx.shape[1]]

        tokens_per_expert_list = tokens_per_expert.cpu().tolist()

        # Run only expert 0
        num_tokens_0 = int(tokens_per_expert_list[0])
        if num_tokens_0 > 0:
            tokens_for_expert_0 = sorted_tokens[:num_tokens_0]
            expert_out = moe.experts[0](tokens_for_expert_0)
            return expert_out
        return sorted_tokens[:1]


class MoeStageABAllExpertsPipeline(_BaseMoeInferPipeline):
    """Stage A + B (all experts): Token routing + full expert dispatch loop.

    Runs all 64 experts on their token slices. Outputs list of expert results (cat'd).
    """

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        x, topk_idx, topk_weight, orig_shape = _run_through_layer0_and_layer1_attn_and_gate(
            self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop,
        )
        moe = self.layer1.mlp

        cnts = topk_idx.new_zeros((topk_idx.shape[0], len(moe.experts)))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = x[idxs // topk_idx.shape[1]]

        tokens_per_expert_list = tokens_per_expert.cpu().tolist()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert_list):
            end_idx = start_idx + int(num_tokens)
            if num_tokens == 0:
                continue
            expert = moe.experts[i + moe.ep_rank * moe.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        return outs


class MoeStageABCPipeline(_BaseMoeInferPipeline):
    """Stage A + B + C: Token routing + all experts + output reassembly (cat + unsort)."""

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        x, topk_idx, topk_weight, orig_shape = _run_through_layer0_and_layer1_attn_and_gate(
            self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop,
        )
        moe = self.layer1.mlp

        cnts = topk_idx.new_zeros((topk_idx.shape[0], len(moe.experts)))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = x[idxs // topk_idx.shape[1]]

        tokens_per_expert_list = tokens_per_expert.cpu().tolist()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert_list):
            end_idx = start_idx + int(num_tokens)
            if num_tokens == 0:
                continue
            expert = moe.experts[i + moe.ep_rank * moe.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs

        return new_x


class MoeStageABCDPipeline(_BaseMoeInferPipeline):
    """Stage A + B + C + D: Full moe_infer = token routing + experts + unsort + weighted sum."""

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        x, topk_idx, topk_weight, orig_shape = _run_through_layer0_and_layer1_attn_and_gate(
            self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop,
        )
        moe = self.layer1.mlp

        cnts = topk_idx.new_zeros((topk_idx.shape[0], len(moe.experts)))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = x[idxs // topk_idx.shape[1]]

        tokens_per_expert_list = tokens_per_expert.cpu().tolist()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert_list):
            end_idx = start_idx + int(num_tokens)
            if num_tokens == 0:
                continue
            expert = moe.experts[i + moe.ep_rank * moe.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_idx.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out.view(*orig_shape)


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


def _make_fixture(full_model_and_raw_inputs, pipeline_cls):
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
def stage_a_model(full_model_and_raw_inputs):
    return _make_fixture(full_model_and_raw_inputs, MoeStageAPipeline)


@pytest.fixture(scope="module")
def stage_ab_1expert_model(full_model_and_raw_inputs):
    return _make_fixture(full_model_and_raw_inputs, MoeStageAB1ExpertPipeline)


@pytest.fixture(scope="module")
def stage_ab_all_model(full_model_and_raw_inputs):
    return _make_fixture(full_model_and_raw_inputs, MoeStageABAllExpertsPipeline)


@pytest.fixture(scope="module")
def stage_abc_model(full_model_and_raw_inputs):
    return _make_fixture(full_model_and_raw_inputs, MoeStageABCPipeline)


@pytest.fixture(scope="module")
def stage_abcd_model(full_model_and_raw_inputs):
    return _make_fixture(full_model_and_raw_inputs, MoeStageABCDPipeline)


# ---------------------------------------------------------------------------
# Stage A: Token routing prep (new_zeros, scatter_, sum, argsort, index)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_moe_stage_a(stage_a_model):
    """
    Token routing prep: new_zeros → scatter_ → sum → argsort → sorted_tokens.
    No expert MLPs yet.
    """
    pipeline, inputs = stage_a_model
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Stage A+B (1 expert): routing + single expert MLP
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_moe_stage_ab_1expert(stage_ab_1expert_model):
    """
    Token routing + expert[0] MLP only (1280→896→1280).
    Tests if a single expert with dynamic-size slice hangs.
    """
    pipeline, inputs = stage_ab_1expert_model
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Stage A+B (all): routing + full expert dispatch loop (64 experts)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_moe_stage_ab_all(stage_ab_all_model):
    """
    Token routing + all 64 expert MLPs (top-6 dispatch, dynamic slicing).
    Full expert loop with torch.cat of outputs.
    """
    pipeline, inputs = stage_ab_all_model
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Stage A+B+C: routing + experts + output reassembly (unsort)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_moe_stage_abc(stage_abc_model):
    """
    Token routing + all experts + output reassembly.
    Adds: new_x = empty_like → new_x[idxs] = outs (scatter back to original order).
    """
    pipeline, inputs = stage_abc_model
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Stage A+B+C+D: full moe_infer (routing + experts + unsort + weighted sum)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_moe_stage_abcd(stage_abcd_model):
    """
    Full moe_infer: routing + experts + unsort + weighted sum.
    view → type → mul_(topk_weight) → sum(dim=1) → type.
    Should reproduce the hang from test_step4_moe_infer.
    """
    pipeline, inputs = stage_abcd_model
    run_op_test(pipeline, inputs, framework=Framework.TORCH)
