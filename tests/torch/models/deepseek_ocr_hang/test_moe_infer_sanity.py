# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Consolidated moe_infer sanity tests for DeepSeek OCR hang investigation.

Loads the actual DeepSeek OCR model and wraps different slices of
moe_infer into end-to-end pipeline modules that take the raw model
inputs (input_ids, patches, etc.) and run vision_embed → layer0 →
layer1_attn → layernorm → gate → <moe_infer block> all inside a single
forward pass.  This ensures moe_infer's inputs (x, topk_idx, topk_weight)
are XLA intermediate tensors — exactly matching the real model's graph
structure and reproducing the hang.

moe_infer blocks (modeling_deepseekv2.py lines 360-433, ep_size=1):

  Block A — Token Sort (lines 361-365):
    cnts = new_zeros → scatter_(1, topk_ids, 1) → sum(dim=0)
    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = x[idxs // topk_ids.shape[1]]

  Block B — Graph Break (line 399):
    tokens_per_expert = tokens_per_expert.cpu().tolist()

  Block C — Expert Dispatch (lines 401-411):
    for each expert: slice sorted_tokens → expert MLP → append

  Block D — Cat (line 413):
    outs = torch.cat(outputs)

  Block E — Unsort + Weighted Sum (lines 424-432):
    new_x = empty_like(outs);  new_x[idxs] = outs
    final = new_x.view(*topk_ids.shape, -1).type().mul_().sum(1).type()

Tests (incremental, each adds ops to the previous):
  test_token_sort                — Block A only (no graph break, no experts)
  test_graph_break_only          — Block A + .cpu().tolist() graph break
  test_first_slice_only          — + sorted_tokens[:N] slice for first expert
  test_first_expert_no_cat       — + expert[0] MLP (no zeros, no cat)
  test_first_expert_with_zeros_cat — + zeros for 63 others + cat
  test_expert_dispatch_1         — same as above via generic N-expert pipeline
  test_expert_dispatch_2         — 2 real experts, rest zeros
  test_expert_dispatch_8         — 8 real experts, rest zeros
  test_expert_dispatch_64        — all 64 experts
  test_unsort_weighted_sum       — Blocks A+B + zeros (skip experts) + E
  test_moe_infer_full            — All blocks = full moe_infer
"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask as _prepare_4d_mask,
)


# ---------------------------------------------------------------------------
# Shared: masked_scatter decomp + vision embed + upstream pipeline
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


def _vision_embed_forward(pipeline, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
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
                    wcn, hcn = crop_shape[0], crop_shape[1]

                    gf = global_features.view(h, w, n_dim)
                    gf = torch.cat(
                        [gf, pipeline.image_newline[None, None, :].expand(h, 1, n_dim)],
                        dim=1,
                    ).view(-1, n_dim)

                    lf = (
                        local_features.view(hcn, wcn, h2, w2, n_dim2)
                        .permute(0, 2, 1, 3, 4)
                        .reshape(hcn * h2, wcn * w2, n_dim2)
                    )
                    lf = torch.cat(
                        [lf, pipeline.image_newline[None, None, :].expand(hcn * h2, 1, n_dim2)],
                        dim=1,
                    ).view(-1, n_dim2)

                    glf = torch.cat([lf, gf, pipeline.view_seperator[None, :]], dim=0)
                else:
                    gf1 = sam_model(io)
                    gf2 = vision_model(io, gf1)
                    global_features = torch.cat(
                        (gf2[:, 1:], gf1.flatten(2).permute(0, 2, 1)), dim=-1,
                    )
                    global_features = pipeline.projector(global_features)
                    _, hw, n_dim = global_features.shape
                    h = w = int(hw**0.5)
                    gf = global_features.view(h, w, n_dim)
                    gf = torch.cat(
                        [gf, pipeline.image_newline[None, None, :].expand(h, 1, n_dim)],
                        dim=1,
                    ).view(-1, n_dim)
                    glf = torch.cat([gf, pipeline.view_seperator[None, :]], dim=0)

                images_in_this_batch.append(glf)

            if images_in_this_batch:
                images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                inputs_embeds[idx] = _masked_scatter_decomp_muladd(
                    inputs_embeds[idx], images_seq_mask[idx], images_in_this_batch,
                )
            idx += 1

    return inputs_embeds


def _run_upstream(pipeline, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
    """Run vision_embed → layer0 → layer1_attn → layernorm → gate.

    Returns (moe, x_flat, topk_idx, topk_weight, orig_shape) with all
    tensors as live XLA intermediates (not materialized to CPU).
    """
    inputs_embeds = _vision_embed_forward(
        pipeline, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop,
    )

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

    # Layer 0
    hidden_states = inputs_embeds
    layer_out = pipeline.layer0(
        hidden_states, attention_mask=attention_mask, position_ids=position_ids,
        past_key_value=None, output_attentions=False, use_cache=False,
    )
    hidden_states = layer_out[0]

    # Layer 1 attention half
    layer1 = pipeline.layer1
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

    # Post-attention layernorm + gate
    hidden_states = layer1.post_attention_layernorm(hidden_states)
    moe = layer1.mlp
    orig_shape = hidden_states.shape
    topk_idx, topk_weight, _ = moe.gate(hidden_states)
    x_flat = hidden_states.view(-1, hidden_states.shape[-1])

    return moe, x_flat, topk_idx, topk_weight, orig_shape


# ---------------------------------------------------------------------------
# Pipeline modules — each includes full upstream + a moe_infer slice
# ---------------------------------------------------------------------------

class _BasePipeline(nn.Module):
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


class TokenSortPipeline(_BasePipeline):
    """Block A: upstream → token sort only (no graph break, no experts)."""

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        moe, x, topk_idx, topk_weight, orig_shape = _run_upstream(
            self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop,
        )
        cnts = topk_idx.new_zeros((topk_idx.shape[0], len(moe.experts)))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = x[idxs // topk_idx.shape[1]]
        return sorted_tokens


class GraphBreakOnlyPipeline(_BasePipeline):
    """Block A + graph break: token sort + .cpu().tolist() + return sorted_tokens.

    Tests whether the graph break itself (forcing XLA compilation of the
    upstream + token sort graph) causes the hang.
    """

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        moe, x, topk_idx, topk_weight, orig_shape = _run_upstream(
            self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop,
        )
        cnts = topk_idx.new_zeros((topk_idx.shape[0], len(moe.experts)))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = x[idxs // topk_idx.shape[1]]

        tokens_per_expert = tokens_per_expert.cpu().tolist()

        return sorted_tokens


class FirstSliceOnlyPipeline(_BasePipeline):
    """Block A + graph break + slice: return first expert's token slice.

    Tests whether dynamic slicing after the graph break causes the hang.
    """

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        moe, x, topk_idx, topk_weight, orig_shape = _run_upstream(
            self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop,
        )
        cnts = topk_idx.new_zeros((topk_idx.shape[0], len(moe.experts)))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = x[idxs // topk_idx.shape[1]]

        tokens_per_expert = tokens_per_expert.cpu().tolist()

        first_expert_tokens = int(tokens_per_expert[0])
        return sorted_tokens[:first_expert_tokens]


class FirstExpertNoCatPipeline(_BasePipeline):
    """Block A + graph break + slice + 1 expert MLP (no cat, no zeros).

    Tests whether running a single expert MLP on a dynamic slice hangs.
    Returns just the single expert's output.
    """

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        moe, x, topk_idx, topk_weight, orig_shape = _run_upstream(
            self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop,
        )
        cnts = topk_idx.new_zeros((topk_idx.shape[0], len(moe.experts)))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = x[idxs // topk_idx.shape[1]]

        tokens_per_expert = tokens_per_expert.cpu().tolist()

        first_expert_tokens = int(tokens_per_expert[0])
        tokens_for_expert_0 = sorted_tokens[:first_expert_tokens]
        expert_out = moe.experts[0](tokens_for_expert_0)
        return expert_out


class FirstExpertWithZerosCatPipeline(_BasePipeline):
    """Block A + graph break + 1 expert MLP + zeros for rest + cat.

    Same as ExpertDispatchPipeline(max_experts=1) but explicit for clarity.
    This is the minimal test that should reproduce the hang.
    """

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        moe, x, topk_idx, topk_weight, orig_shape = _run_upstream(
            self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop,
        )
        cnts = topk_idx.new_zeros((topk_idx.shape[0], len(moe.experts)))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = x[idxs // topk_idx.shape[1]]

        tokens_per_expert = tokens_per_expert.cpu().tolist()

        print(f"\n[DEBUG] sorted_tokens.shape={sorted_tokens.shape}, "
              f"sorted_tokens.device={sorted_tokens.device}")
        print(f"[DEBUG] tokens_per_expert (from cpu): {tokens_per_expert}")
        print(f"[DEBUG] total tokens expected: {sum(tokens_per_expert)}, "
              f"non-zero experts: {sum(1 for t in tokens_per_expert if t > 0)}")

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            print(f"[DEBUG] expert {i}: num_tokens={num_tokens}, "
                  f"slice=[{start_idx}:{end_idx}]", flush=True)
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            print(f"[DEBUG] expert {i}: sliced shape={tokens_for_this_expert.shape}, "
                  f"device={tokens_for_this_expert.device}", flush=True)
            if i == 0:
                expert_out = moe.experts[0](tokens_for_this_expert)
                print(f"[DEBUG] expert {i}: MLP output shape={expert_out.shape}", flush=True)
            else:
                expert_out = torch.zeros_like(tokens_for_this_expert)
                print(f"[DEBUG] expert {i}: zeros_like shape={expert_out.shape}", flush=True)
            outputs.append(expert_out)
            print(f"[DEBUG] expert {i}: appended, total outputs so far={len(outputs)}",
                  flush=True)
            start_idx = end_idx

        print(f"[DEBUG] loop done, {len(outputs)} outputs, about to cat...", flush=True)
        result = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
        print(f"[DEBUG] cat done, result.shape={result.shape}", flush=True)
        return result


class LoopLimitPipeline(_BasePipeline):
    """Block A + graph break + loop limited to max_iters iterations + cat.

    Runs the expert loop but breaks after max_iters non-zero experts,
    using zeros_like for all of them (no real expert MLPs).
    Controls the number of slice + zeros_like ops in the XLA graph.
    """

    def __init__(self, ocr_model, max_iters):
        super().__init__(ocr_model)
        self.max_iters = max_iters

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        moe, x, topk_idx, topk_weight, orig_shape = _run_upstream(
            self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop,
        )
        cnts = topk_idx.new_zeros((topk_idx.shape[0], len(moe.experts)))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = x[idxs // topk_idx.shape[1]]

        tokens_per_expert = tokens_per_expert.cpu().tolist()

        outputs = []
        start_idx = 0
        iters_done = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            if iters_done >= self.max_iters:
                break
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = torch.zeros_like(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx
            iters_done += 1

        return torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)


class MlpPlusZerosLoopPipeline(_BasePipeline):
    """Block A + graph break + 1 expert MLP + N-1 zeros_like iterations + cat.

    First non-zero expert runs the real MLP, remaining iterations up to
    max_iters use zeros_like. Tests the combination of MLP + zeros in
    one XLA graph at varying scales.
    """

    def __init__(self, ocr_model, max_iters):
        super().__init__(ocr_model)
        self.max_iters = max_iters

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        moe, x, topk_idx, topk_weight, orig_shape = _run_upstream(
            self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop,
        )
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
            if iters_done >= self.max_iters:
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


class ExpertDispatchPipeline(_BasePipeline):
    """Blocks A+B+C+D: upstream → token sort → graph break → N experts → cat."""

    def __init__(self, ocr_model, max_experts):
        super().__init__(ocr_model)
        self.max_experts = max_experts

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        moe, x, topk_idx, topk_weight, orig_shape = _run_upstream(
            self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop,
        )
        cnts = topk_idx.new_zeros((topk_idx.shape[0], len(moe.experts)))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = x[idxs // topk_idx.shape[1]]

        tokens_per_expert = tokens_per_expert.cpu().tolist()

        outputs = []
        start_idx = 0
        experts_run = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            if experts_run < self.max_experts:
                expert = moe.experts[i + moe.ep_rank * moe.experts_per_rank]
                expert_out = expert(tokens_for_this_expert)
            else:
                expert_out = torch.zeros_like(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx
            experts_run += 1

        return torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)


class UnsortWeightedSumPipeline(_BasePipeline):
    """Blocks A+B + zeros (skip experts) + E: unsort + weighted sum."""

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        moe, x, topk_idx, topk_weight, orig_shape = _run_upstream(
            self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop,
        )
        cnts = topk_idx.new_zeros((topk_idx.shape[0], len(moe.experts)))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = x[idxs // topk_idx.shape[1]]

        outs = torch.zeros_like(sorted_tokens)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_idx.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


class MoeInferFullPipeline(_BasePipeline):
    """All blocks A+B+C+D+E: exact copy of moe_infer (ep_size=1)."""

    def forward(self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop):
        moe, x, topk_idx, topk_weight, orig_shape = _run_upstream(
            self, input_ids, patches, image_ori, images_seq_mask, images_spatial_crop,
        )
        y = moe.moe_infer(x, topk_idx, topk_weight).view(*orig_shape)
        return (y,)


# ---------------------------------------------------------------------------
# Model loading + input extraction
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


def _extract_inputs(raw_inputs):
    return [
        raw_inputs["input_ids"],
        raw_inputs["images"][0][0],
        raw_inputs["images"][0][1],
        raw_inputs["images_seq_mask"],
        raw_inputs["images_spatial_crop"],
    ]


def _make_pipeline(full_model, pipeline_cls, **kwargs):
    pipeline = pipeline_cls(full_model, **kwargs)
    pipeline.eval()
    pipeline = pipeline.to(torch.bfloat16)
    return pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_model_and_inputs():
    full_model, raw_inputs = _load_model_and_inputs()
    return full_model, _extract_inputs(raw_inputs)


# ---------------------------------------------------------------------------
# Test 1: Token Sort (Block A)
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_token_sort(full_model_and_inputs):
    """upstream → Block A: new_zeros → scatter_ → sum → argsort → fancy index.
    No graph break, no experts.
    """
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, TokenSortPipeline)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 1b: Token Sort + graph break (force XLA compilation)
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_graph_break_only(full_model_and_inputs):
    """upstream → Block A + .cpu().tolist() graph break → return sorted_tokens.
    If this hangs: the XLA compilation of upstream+token_sort graph is the issue.
    If this passes: something after the graph break causes the hang.
    """
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, GraphBreakOnlyPipeline)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 1c: Token Sort + graph break + first slice
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_first_slice_only(full_model_and_inputs):
    """upstream → Block A + graph break + sorted_tokens[:N] for first expert.
    If this hangs: dynamic slicing after graph break is the issue.
    If this passes: the expert MLP or cat causes the hang.
    """
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, FirstSliceOnlyPipeline)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 1d: Token Sort + graph break + first slice + 1 expert MLP
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_first_expert_no_cat(full_model_and_inputs):
    """upstream → Block A + graph break + slice + expert[0] MLP.
    Returns only the single expert's output (no zeros, no cat).
    If this hangs: the expert MLP on a dynamic slice is the issue.
    If this passes: the zeros+cat for remaining experts causes the hang.
    """
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, FirstExpertNoCatPipeline)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 1e: Token Sort + graph break + 1 expert MLP + zeros for rest + cat
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_first_expert_with_zeros_cat(full_model_and_inputs):
    """upstream → Block A + graph break + 1 expert + zeros for 63 others + cat.
    Same logic as test_expert_dispatch_1 but hardcoded to expert[0].
    If this hangs: the loop with zeros_like + cat is the issue.
    If this passes: should match test_expert_dispatch_1 behavior.
    """
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, FirstExpertWithZerosCatPipeline)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 1f: Loop with 4 iterations (zeros only, no expert MLP)
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_loop_4_iters(full_model_and_inputs):
    """upstream → Block A + graph break + 4 slice+zeros_like iterations + cat.
    If this hangs: 4 dynamic slices are enough to break PJRT compilation.
    If this passes: need more iterations to trigger the hang.
    """
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, LoopLimitPipeline, max_iters=4)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 1g: Loop with 8 iterations (zeros only, no expert MLP)
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_loop_8_iters(full_model_and_inputs):
    """upstream → Block A + graph break + 8 slice+zeros_like iterations + cat.
    If this hangs: 8 dynamic slices are enough to break PJRT compilation.
    If this passes: need more iterations to trigger the hang.
    """
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, LoopLimitPipeline, max_iters=8)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 1h: Loop with 16 iterations (zeros only, no expert MLP)
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_loop_16_iters(full_model_and_inputs):
    """upstream → Block A + graph break + 16 slice+zeros_like iterations + cat."""
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, LoopLimitPipeline, max_iters=16)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 1i: Loop with 32 iterations (zeros only, no expert MLP)
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_loop_32_iters(full_model_and_inputs):
    """upstream → Block A + graph break + 32 slice+zeros_like iterations + cat."""
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, LoopLimitPipeline, max_iters=32)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 1j: 1 MLP + 3 zeros (4 total iterations)
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_mlp_plus_zeros_4(full_model_and_inputs):
    """upstream → Block A + graph break + 1 MLP + 3 zeros_like + cat (4 iters)."""
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, MlpPlusZerosLoopPipeline, max_iters=4)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 1k: 1 MLP + 7 zeros (8 total iterations)
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_mlp_plus_zeros_8(full_model_and_inputs):
    """upstream → Block A + graph break + 1 MLP + 7 zeros_like + cat (8 iters)."""
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, MlpPlusZerosLoopPipeline, max_iters=8)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 1l: 1 MLP + 15 zeros (16 total iterations)
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_mlp_plus_zeros_16(full_model_and_inputs):
    """upstream → Block A + graph break + 1 MLP + 15 zeros_like + cat (16 iters)."""
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, MlpPlusZerosLoopPipeline, max_iters=16)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 1m: 1 MLP + 31 zeros (32 total iterations)
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_mlp_plus_zeros_32(full_model_and_inputs):
    """upstream → Block A + graph break + 1 MLP + 31 zeros_like + cat (32 iters)."""
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, MlpPlusZerosLoopPipeline, max_iters=32)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 1n: 1 MLP + 47 zeros (48 total iterations)
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_mlp_plus_zeros_48(full_model_and_inputs):
    """upstream → Block A + graph break + 1 MLP + 47 zeros_like + cat (48 iters)."""
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, MlpPlusZerosLoopPipeline, max_iters=48)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 1o: 1 MLP + 63 zeros (64 total iterations)
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_mlp_plus_zeros_64(full_model_and_inputs):
    """upstream → Block A + graph break + 1 MLP + 63 zeros_like + cat (64 iters).
    Should reproduce the hang from test_first_expert_with_zeros_cat.
    """
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, MlpPlusZerosLoopPipeline, max_iters=64)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 1o: Loop with 48 iterations (zeros only, no expert MLP)
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_loop_48_iters(full_model_and_inputs):
    """upstream → Block A + graph break + 48 slice+zeros_like iterations + cat."""
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, LoopLimitPipeline, max_iters=48)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 1k: Loop with 64 iterations (zeros only, no expert MLP)
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_loop_64_iters(full_model_and_inputs):
    """upstream → Block A + graph break + 64 slice+zeros_like iterations + cat.
    Same iteration count as the full model (all 64 experts get tokens on TT).
    """
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, LoopLimitPipeline, max_iters=64)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 2: Expert Dispatch — 1 expert
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_expert_dispatch_1(full_model_and_inputs):
    """upstream → Blocks A+B+C+D: token sort + 1 real expert + 63 zeros + cat."""
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, ExpertDispatchPipeline, max_experts=1)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 3: Expert Dispatch — 2 experts
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_expert_dispatch_2(full_model_and_inputs):
    """upstream → Blocks A+B+C+D: token sort + 2 real experts + 62 zeros + cat."""
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, ExpertDispatchPipeline, max_experts=2)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 4: Expert Dispatch — 8 experts
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_expert_dispatch_8(full_model_and_inputs):
    """upstream → Blocks A+B+C+D: token sort + 8 real experts + 56 zeros + cat."""
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, ExpertDispatchPipeline, max_experts=8)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 5: Expert Dispatch — all 64 experts
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_expert_dispatch_64(full_model_and_inputs):
    """upstream → Blocks A+B+C+D: token sort + all 64 experts + cat."""
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, ExpertDispatchPipeline, max_experts=64)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 6: Unsort + Weighted Sum (Block E)
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_unsort_weighted_sum(full_model_and_inputs):
    """upstream → Blocks A+B + zeros (skip experts) + E: unsort + weighted sum."""
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, UnsortWeightedSumPipeline)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Test 7: Full moe_infer
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_moe_infer_full(full_model_and_inputs):
    """upstream → all blocks A+B+C+D+E via moe.moe_infer (ep_size=1)."""
    full_model, inputs = full_model_and_inputs
    pipeline = _make_pipeline(full_model, MoeInferFullPipeline)
    run_op_test(pipeline, inputs, framework=Framework.TORCH)
