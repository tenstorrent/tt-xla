# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
moe_infer bisection tests for DeepSeek OCR hang investigation.

test_step4_moe_infer (from test_deepseek_ocr_layer1_bisect.py) hangs.
This file bisects moe_infer by using modified versions of the function
that stop at different points, while respecting the natural graph break
at tokens_per_expert.cpu().tolist() (line 399 of modeling_deepseekv2.py).

moe_infer internal flow (modeling_deepseekv2.py lines 360-433):
  --- graph segment 1 (before .cpu() graph break) ---
  A. Token infrastructure: new_zeros → scatter_ → sum → argsort → index
  --- graph break: tokens_per_expert.cpu().tolist() ---
  --- graph segment 2 (per-expert, Python loop) ---
  B. Expert dispatch: for each of 64 experts: slice → expert MLP → append
  C. Cat outputs: torch.cat(outputs)
  --- graph segment 3 ---
  D. Unsort + weight: empty_like → index_assign → view → type → mul_ → sum

Returning intermediate tensors from segment A breaks XLA compilation (MHLO
conversion fails). So instead we create modified moe_infer functions that
run the graph-break-safe portion and stop at different points:

  test_moe_experts_only       — full expert loop + cat, skip unsort+weight
  test_moe_unsort_weight_only — experts produce zeros, only unsort+weight runs
  test_moe_full               — full moe_infer (= test_step4, expected to HANG)
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
# Modified moe_infer variants (respect graph break at .cpu().tolist())
# ---------------------------------------------------------------------------

def _moe_infer_experts_only(moe, x, topk_ids, topk_weight):
    """Full expert loop + cat but skip unsort+weight (lines 361-413)."""
    cnts = topk_ids.new_zeros((topk_ids.shape[0], len(moe.experts)))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = x[idxs // topk_ids.shape[1]]

    tokens_per_expert = tokens_per_expert.cpu().tolist()

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        expert = moe.experts[i + moe.ep_rank * moe.experts_per_rank]
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        expert_out = expert(tokens_for_this_expert)
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    return outs


def _moe_infer_unsort_weight_only(moe, x, topk_ids, topk_weight):
    """Token sort + experts produce zeros + unsort+weight (lines 361-433).

    Expert MLPs are replaced with zeros to isolate the unsort+weight ops.
    """
    cnts = topk_ids.new_zeros((topk_ids.shape[0], len(moe.experts)))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = x[idxs // topk_ids.shape[1]]

    outs = torch.zeros_like(sorted_tokens)

    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    final_out = (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    return final_out


# ---------------------------------------------------------------------------
# Pipeline wrappers
# ---------------------------------------------------------------------------

class _BaseMoeInferBisectPipeline(nn.Module):
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


class MoeExpertsOnlyPipeline(_BaseMoeInferBisectPipeline):
    """Step A: ... + gate + token sort + all 64 expert MLPs + cat (skip unsort+weight).

    Tests: token infrastructure (scatter_, sum, argsort, index) across graph break
    + full expert dispatch loop (64 x DeepseekV2MLP) + cat.
    If this hangs → problem is in expert dispatch or token sort.
    If this passes → problem is in unsort+weight ops.
    """

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        inputs_embeds = _vision_embed_forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop)
        attention_mask, position_ids = _pre_decoder_setup(self, inputs_embeds)
        hidden_states = _run_layer0(self, inputs_embeds, attention_mask, position_ids)
        hidden_states = _run_layer1_attn_half(self, hidden_states, attention_mask, position_ids)
        moe, hidden_states, topk_idx, topk_weight, orig_shape = _run_through_gate(self, hidden_states)

        outs = _moe_infer_experts_only(moe, hidden_states, topk_idx, topk_weight)
        return (outs,)


class MoeUnsortWeightOnlyPipeline(_BaseMoeInferBisectPipeline):
    """Step B: ... + gate + token sort + zeros (no experts) + unsort+weight.

    Expert MLPs replaced with zeros to isolate unsort+weight ops:
    empty_like → index_assign → view → type → mul_ → sum → type
    If this hangs → problem is in the unsort+weight ops.
    If this passes → problem is in the expert MLPs themselves.
    """

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        inputs_embeds = _vision_embed_forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop)
        attention_mask, position_ids = _pre_decoder_setup(self, inputs_embeds)
        hidden_states = _run_layer0(self, inputs_embeds, attention_mask, position_ids)
        hidden_states = _run_layer1_attn_half(self, hidden_states, attention_mask, position_ids)
        moe, hidden_states, topk_idx, topk_weight, orig_shape = _run_through_gate(self, hidden_states)

        y = _moe_infer_unsort_weight_only(moe, hidden_states, topk_idx, topk_weight).view(*orig_shape)
        return (y,)


class MoeFullPipeline(_BaseMoeInferBisectPipeline):
    """Step C: ... + gate + full moe_infer (= test_step4_moe_infer).

    Calls moe.moe_infer directly — expected to HANG (>5min).
    """

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        inputs_embeds = _vision_embed_forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop)
        attention_mask, position_ids = _pre_decoder_setup(self, inputs_embeds)
        hidden_states = _run_layer0(self, inputs_embeds, attention_mask, position_ids)
        hidden_states = _run_layer1_attn_half(self, hidden_states, attention_mask, position_ids)
        moe, hidden_states, topk_idx, topk_weight, orig_shape = _run_through_gate(self, hidden_states)

        y = moe.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        return (y,)


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
def step_a_model(full_model_and_raw_inputs):
    return _make_bisect_fixture(full_model_and_raw_inputs, MoeExpertsOnlyPipeline)


@pytest.fixture(scope="module")
def step_b_model(full_model_and_raw_inputs):
    return _make_bisect_fixture(full_model_and_raw_inputs, MoeUnsortWeightOnlyPipeline)


@pytest.fixture(scope="module")
def step_c_model(full_model_and_raw_inputs):
    return _make_bisect_fixture(full_model_and_raw_inputs, MoeFullPipeline)


# ---------------------------------------------------------------------------
# Step A: all 64 experts + cat (skip unsort+weight)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_moe_experts_only(step_a_model):
    """
    ... + gate + token sort + 64 expert MLPs + cat.
    Skip unsort+weight — isolates expert dispatch from final assembly.
    """
    pipeline, inputs = step_a_model
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Step B: token sort + zeros (no experts) + unsort+weight
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_moe_unsort_weight_only(step_b_model):
    """
    ... + gate + token sort + zeros (experts skipped) + unsort+weight.
    Isolates: empty_like → index_assign → view → type → mul_ → sum.
    """
    pipeline, inputs = step_b_model
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Step C: full moe_infer (reference — expected to HANG)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_moe_full(step_c_model):
    """
    ... + gate + full moe_infer (token sort + 64 experts + unsort + weighted sum).
    Same as test_step4_moe_infer — expected to HANG (>5min).
    """
    pipeline, inputs = step_c_model
    run_op_test(pipeline, inputs, framework=Framework.TORCH)
