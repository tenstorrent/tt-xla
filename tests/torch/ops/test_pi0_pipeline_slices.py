# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
PI0 pipeline slices in **the same order as** ``PI0Pytorch.sample_actions`` (lerobot
``modeling_pi0``), using ``run_op_test`` to see which stage first fails on TT.

Order (see upstream ``sample_actions`` / ``denoise_step``):
  1. ``embed_prefix`` — vision + language embeddings, ``torch.cat`` on dim=1.
  2. Prefix ``paligemma_with_expert.forward`` with ``inputs_embeds=[prefix, None]``,
     ``use_cache=True`` — builds KV cache (large pytree output). If this fails on TT
     with SHLO ``reduce_window`` / compile error 13, replication is still aligned
     with ``sample_actions``; use **stage 2a** (same tensors up to ``_prepare_attention_masks_4d``)
     to validate the prefix prep subgraph on device.
  2a. Prefix attention mask + position ids only (``sample_actions`` lines before ``forward``).
  2b. ``denoise_step`` suffix ``position_ids`` fragment (offset + ``cumsum`` on suffix pads).
  3. ``embed_suffix`` — state + action/time MLP path (includes ``torch.cat`` on dim=2).
  4. **Mask concat slice** — only ``torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)``
     as in ``denoise_step`` (this is the rank-3 concat along the last axis).

These tests are **heavy** (HF weights + LeRobot dataset), same cost class as the
full Pi0 model test.

Upstream ``PI0Pytorch`` / ``make_att_2d_masks`` / ``sample_actions`` live in the
**lerobot** package (not in this repo), module path ``lerobot.policies.pi0.modeling_pi0``.
On disk that is typically
``.../site-packages/lerobot/policies/pi0/modeling_pi0.py``; resolve with
``python -c "import lerobot.policies.pi0.modeling_pi0 as m; print(m.__file__)"``.

Uses ``tests.runner.requirements.RequirementsManager`` for the Pi0 loader (same as
``tests/runner/test_models.py``) so per-model requirements apply during load.

``third_party/tt_forge_models/pi_0/pytorch/src/model.py`` temporarily replaces
``torch.cumsum`` during ``policy.forward`` (bool → ``torch.long`` before accumulation).
Context manager ``pi0_pipeline_shared.pi0_torch_cumsum_patch_like_model_py_forward`` copies that behavior
for prefix tensors, stage-4 ``make_att_2d_masks``, and stage-2b suffix ``position_ids``.

Requires libero dataset / HF access like the full Pi0 model test.

Incremental prefix block **815--845**: **test_pi0_sample_actions_incremental_block_815_845**.

Loop-length ``sample_actions`` sanities (``num_steps`` = 1, 2, or full config) live in
**``test_pi0_sample_actions_num_steps.py``**.

Also: **test_pi0_sample_actions_incremental_full_steps_plus_policy_crop**,
**test_pi0_policy_forward_like_custom_model_py**; CPU checks
**test_pi0_incremental_pipeline_matches_core_sample_actions_cpu**,
**test_pi0_policy_forward_matches_first_cropped_timestep_cpu**.
"""

from __future__ import annotations

import pytest
import torch

from infra import Framework, run_op_test
from tests.torch.ops.pi0_pipeline_shared import (
    Pi0IncrementalFullStepsPlusPolicyCrop,
    Pi0PolicyForwardLikeCustomModelPy,
    Pi0SampleActionsBlock815to845,
    clear_policy_device_queue,
    crop_actions_chunk_like_model_py,
    pi0_torch_cumsum_patch_like_model_py_forward,
    prefix_tensors_before_paligemma_forward,
    sample_actions_lines_815_884,
    set_paligemma_lm_attn_eager,
)
from utils import Category


# --- Stage 1: embed_prefix (matches start of sample_actions) ---


class Stage01_EmbedPrefixEmbs(torch.nn.Module):
    """Return only prefix token embeddings (single tensor) for stable compare."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

    def forward(self, images, img_masks, lang_tokens, lang_masks):
        embs, _pad, _att = self.core.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        return embs


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_stage01_embed_prefix(request, pi0_bundle):
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        _state,
        _noise,
    ) = pi0_bundle
    run_op_test(
        Stage01_EmbedPrefixEmbs(core),
        [images, img_masks, lang_tokens, lang_masks],
        framework=Framework.TORCH,
        request=request,
    )


# --- Stage 2: prefix forward + KV cache (matches sample_actions after embed_prefix) ---


class Stage02_PrefixForwardCache(torch.nn.Module):
    """Prefix-only paligemma forward; returns ``past_key_values`` pytree."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

    def forward(self, images, img_masks, lang_tokens, lang_masks):
        m = self.core
        prefix_embs, prefix_att_2d_masks_4d, prefix_position_ids = (
            prefix_tensors_before_paligemma_forward(
                m, images, img_masks, lang_tokens, lang_masks
            )
        )
        set_paligemma_lm_attn_eager(m.paligemma_with_expert.paligemma)
        _out, past_key_values = m.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return past_key_values


class Stage02a_PrefixAttentionPrep(torch.nn.Module):
    """``make_att_2d_masks`` → ``cumsum`` position ids → ``_prepare_attention_masks_4d`` only."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

    def forward(self, images, img_masks, lang_tokens, lang_masks):
        _embs, att_4d, pos_ids = prefix_tensors_before_paligemma_forward(
            self.core, images, img_masks, lang_tokens, lang_masks
        )
        return att_4d, pos_ids.to(torch.float32)


class Stage02b_DenoisePositionIds(torch.nn.Module):
    """``denoise_step`` ``position_ids`` fragment; suffix ``cumsum`` under policy patch."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        _pe, prefix_pad_masks, _pa = m.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        bsize = state.shape[0]
        device = state.device
        timestep = torch.tensor(0.85, dtype=torch.float32, device=device).expand(bsize)
        _se, suffix_pad_masks, _sa, _ada = m.embed_suffix(state, noise, timestep)
        with pi0_torch_cumsum_patch_like_model_py_forward():
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        return position_ids.to(torch.float32)


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_stage02a_prefix_attention_prep(request, pi0_bundle):
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        _state,
        _noise,
    ) = pi0_bundle
    run_op_test(
        Stage02a_PrefixAttentionPrep(core),
        [images, img_masks, lang_tokens, lang_masks],
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_stage02b_denoise_position_ids(request, pi0_bundle):
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    run_op_test(
        Stage02b_DenoisePositionIds(core),
        [images, img_masks, lang_tokens, lang_masks, state, noise],
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_stage02_prefix_kv_cache(request, pi0_bundle):
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        _state,
        _noise,
    ) = pi0_bundle
    run_op_test(
        Stage02_PrefixForwardCache(core),
        [images, img_masks, lang_tokens, lang_masks],
        framework=Framework.TORCH,
        request=request,
    )


# --- Stage 3: embed_suffix (first part of each denoise_step) ---


class Stage03_EmbedSuffixEmbs(torch.nn.Module):
    """Suffix embeddings tensor after internal ``torch.cat`` (dim=1)."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

    def forward(self, state, noise, timestep_1d):
        embs, _pad, _att, _ada = self.core.embed_suffix(state, noise, timestep_1d)
        return embs


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_stage03_embed_suffix(request, pi0_bundle):
    core, _policy, (
        _images,
        _img_masks,
        _lang_tokens,
        _lang_masks,
        state,
        noise,
    ) = pi0_bundle
    bsize = state.shape[0]
    device = state.device
    timestep = torch.tensor(0.85, dtype=torch.float32, device=device).expand(bsize)
    run_op_test(
        Stage03_EmbedSuffixEmbs(core),
        [state, noise, timestep],
        framework=Framework.TORCH,
        request=request,
    )


# --- Stage 4: attention-mask concat inside denoise_step (dim=2 on rank-3 tensors) ---


class Stage04_DenoiseAttentionMaskConcat(torch.nn.Module):
    """
    Replays only::

        full_att_2d_masks = torch.cat(
            [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
        )

    from ``PI0Pytorch.denoise_step``, using **live** prefix/suffix masks from the
    same tensors as the full model (matches shapes from a real step).

    ``make_att_2d_masks`` runs under ``pi0_torch_cumsum_patch_like_model_py_forward``
    (same as ``pi_0/pytorch/src/model.py`` policy ``forward``).
    """

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        from lerobot.policies.pi0 import modeling_pi0

        make_att_2d_masks = modeling_pi0.make_att_2d_masks
        m = self.core
        _prefix_embs, prefix_pad_masks, _prefix_att = m.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        bsize = state.shape[0]
        device = state.device
        timestep = torch.tensor(0.85, dtype=torch.float32, device=device).expand(
            bsize
        )
        _suffix_embs, suffix_pad_masks, suffix_att_masks, _ada = m.embed_suffix(
            state, noise, timestep
        )
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        with pi0_torch_cumsum_patch_like_model_py_forward():
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        return torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_stage04_denoise_mask_concat(request, pi0_bundle):
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    run_op_test(
        Stage04_DenoiseAttentionMaskConcat(core),
        [images, img_masks, lang_tokens, lang_masks, state, noise],
        framework=Framework.TORCH,
        request=request,
    )


# --- Incremental prefix KV (~815--845); denoise loop length tests: ``test_pi0_sample_actions_num_steps.py`` ---


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_sample_actions_incremental_block_815_845(request, pi0_bundle):
    """Test1: same subgraph as ``sample_actions`` through prefix KV (``modeling_pi0`` ~815--845)."""
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    run_op_test(
        Pi0SampleActionsBlock815to845(core),
        [images, img_masks, lang_tokens, lang_masks, state, noise],
        framework=Framework.TORCH,
        request=request,
    )


def test_pi0_incremental_pipeline_matches_core_sample_actions_cpu(pi0_bundle):
    """Duplicated ``sample_actions_lines_815_884`` matches ``core.sample_actions`` on CPU."""
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    incremental = sample_actions_lines_815_884(
        core,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
        num_steps=None,
        kwargs={},
    )
    with pi0_torch_cumsum_patch_like_model_py_forward():
        reference = core.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )
    torch.testing.assert_close(incremental, reference, rtol=1e-5, atol=1e-5)


def test_pi0_policy_forward_matches_first_cropped_timestep_cpu(pi0_bundle):
    """``policy.forward`` first output matches first timestep of cropped ``sample_actions``."""
    core, policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    with pi0_torch_cumsum_patch_like_model_py_forward():
        raw = core.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )
    cropped = crop_actions_chunk_like_model_py(policy, raw)
    clear_policy_device_queue(policy, state.device)
    first_forward = policy.forward(
        images, img_masks, lang_tokens, lang_masks, state, noise=noise
    )
    expected_first = cropped[:, 0, :]
    torch.testing.assert_close(first_forward, expected_first, rtol=1e-5, atol=1e-5)


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_sample_actions_incremental_full_steps_plus_policy_crop(request, pi0_bundle):
    """Full diffusion + ``model.py`` action dim / horizon crop (still a tensor chunk)."""
    core, policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    run_op_test(
        Pi0IncrementalFullStepsPlusPolicyCrop(core, policy),
        [images, img_masks, lang_tokens, lang_masks, state, noise],
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_policy_forward_like_custom_model_py(request, pi0_bundle):
    """End-to-end customized ``policy.forward`` (queue + first action), runner-style."""
    _core, policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    run_op_test(
        Pi0PolicyForwardLikeCustomModelPy(policy),
        [images, img_masks, lang_tokens, lang_masks, state, noise],
        framework=Framework.TORCH,
        request=request,
    )
