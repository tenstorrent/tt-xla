# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end **chain** bisect for ``num_steps == 2`` on TT.

If ``test_pi0_second_denoise_bisect.py`` passes while ``Pi0SampleActionsBlock815to884NumSteps``
(``num_steps=2``) fails, the bug is likely tied to **one compiled graph** that runs prefix,
**first** ``denoise_step`` on device, then the **second** denoise—rather than to any slice
that skips the first denoise or uses a CPU-frozen ``x_t`` between steps.

Each module here takes **only** the real bundle tensors (same inputs as full
``sample_actions``). Inside ``forward`` it runs, in one graph:

1. Prefix KV (``embed_prefix`` → ``paligemma_with_expert.forward`` with cache).
2. First denoise at ``t = 1.0`` (for ``num_steps=2``, ``dt = -0.5``) and ``x_t`` update.
3. An increasing prefix of the **second** denoise (``t = 0.5``), stopping at a named boundary.

Run tests in numeric order; the **first** failure localizes the smallest *chained* subgraph
that still reproduces the TT issue. The last test is the full two-step body (same as
``test_pi0_sample_actions_num_steps`` two-step) as a baseline.
"""

from __future__ import annotations

import pytest
import torch

from infra import Framework, run_op_test
from tests.torch.ops.pi0_pipeline_shared import (
    Pi0SampleActionsBlock815to884NumSteps,
    pi0_torch_cumsum_patch_like_model_py_forward,
    set_paligemma_lm_attn_eager,
)
from utils import Category


def _set_gemma_expert_attn_eager(m: torch.nn.Module) -> None:
    exp = m.paligemma_with_expert.gemma_expert
    exp.model.config._attn_implementation = "eager"  # noqa: SLF001


def _prefix_pad_masks_and_pkv(
    m: torch.nn.Module,
    images,
    img_masks,
    lang_tokens,
    lang_masks,
):
    from lerobot.policies.pi0 import modeling_pi0

    make_att_2d_masks = modeling_pi0.make_att_2d_masks
    prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = m._prepare_attention_masks_4d(prefix_att_2d_masks)
    set_paligemma_lm_attn_eager(m.paligemma_with_expert.paligemma)
    _, past_key_values = m.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return prefix_pad_masks, past_key_values


def _prefix_first_denoise_then_state(
    m: torch.nn.Module,
    images,
    img_masks,
    lang_tokens,
    lang_masks,
    state,
    noise: torch.Tensor,
    num_steps: int = 2,
):
    """Prefix + first ``denoise_step`` + ``x_t`` update; returns masks, pkv, ``x_t``, ``time1``."""
    bsize = state.shape[0]
    device = state.device
    dt = -1.0 / num_steps
    prefix_pad_masks, past_key_values = _prefix_pad_masks_and_pkv(
        m, images, img_masks, lang_tokens, lang_masks
    )
    x_t = noise
    time0 = torch.tensor(1.0 + 0 * dt, dtype=torch.float32, device=device).expand(bsize)
    _set_gemma_expert_attn_eager(m)
    v0 = m.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time0)
    x_t = x_t + dt * v0
    time1 = torch.tensor(1.0 + 1 * dt, dtype=torch.float32, device=device).expand(bsize)
    return prefix_pad_masks, past_key_values, x_t, time1


def _bundle_inputs(pi0_bundle):
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    return core, [images, img_masks, lang_tokens, lang_masks, state, noise]


# --- Chains: prefix + 1st denoise on TT, then 2nd-denoise prefix ---


class FullChain01_SecondEmbedSuffix(torch.nn.Module):
    """After first denoise on device: ``embed_suffix`` only (second iteration)."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self.num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            _ppm, _pkv, x_t, time1 = _prefix_first_denoise_then_state(
                m, images, img_masks, lang_tokens, lang_masks, state, noise, self.num_steps
            )
            del _ppm, _pkv
            suffix_embs, _p, _a, _ad = m.embed_suffix(state, x_t, time1)
        return suffix_embs


class FullChain02_SecondThroughRank3Cat(torch.nn.Module):
    """Through ``torch.cat`` of prefix pad 2d + suffix att 2d (dim=2)."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self.num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        from lerobot.policies.pi0 import modeling_pi0

        make_att_2d_masks = modeling_pi0.make_att_2d_masks
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            prefix_pad_masks, _pkv, x_t, time1 = _prefix_first_denoise_then_state(
                m, images, img_masks, lang_tokens, lang_masks, state, noise, self.num_steps
            )
            _e, suffix_pad_masks, suffix_att_masks, _ad = m.embed_suffix(
                state, x_t, time1
            )
            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        return full.to(dtype=torch.float32)


class FullChain03_SecondThroughPositionIds(torch.nn.Module):
    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self.num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        from lerobot.policies.pi0 import modeling_pi0

        make_att_2d_masks = modeling_pi0.make_att_2d_masks
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            prefix_pad_masks, _pkv, x_t, time1 = _prefix_first_denoise_then_state(
                m, images, img_masks, lang_tokens, lang_masks, state, noise, self.num_steps
            )
            _e, suffix_pad_masks, suffix_att_masks, _ad = m.embed_suffix(
                state, x_t, time1
            )
            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d = torch.cat(
                [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
            )
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        return position_ids.to(dtype=torch.float32) + 1e-9 * full_att_2d.to(
            dtype=torch.float32
        ).mean()


class FullChain04_SecondThroughAttnMask4d(torch.nn.Module):
    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self.num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        from lerobot.policies.pi0 import modeling_pi0

        make_att_2d_masks = modeling_pi0.make_att_2d_masks
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            prefix_pad_masks, _pkv, x_t, time1 = _prefix_first_denoise_then_state(
                m, images, img_masks, lang_tokens, lang_masks, state, noise, self.num_steps
            )
            _e, suffix_pad_masks, suffix_att_masks, _ad = m.embed_suffix(
                state, x_t, time1
            )
            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d = torch.cat(
                [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
            )
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            full_4d = m._prepare_attention_masks_4d(full_att_2d)
        return full_4d.to(dtype=torch.float32) + 1e-9 * position_ids.to(
            dtype=torch.float32
        ).mean()


class FullChain05_SecondFullDenoiseStep(torch.nn.Module):
    """Prefix + first denoise + full second ``denoise_step`` (matches inner loop twice)."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self.num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            prefix_pad_masks, past_key_values, x_t, time1 = _prefix_first_denoise_then_state(
                m, images, img_masks, lang_tokens, lang_masks, state, noise, self.num_steps
            )
            _set_gemma_expert_attn_eager(m)
            return m.denoise_step(
                state, prefix_pad_masks, past_key_values, x_t, time1
            )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_full_chain_second_01_embed_suffix(request, pi0_bundle):
    core, args = _bundle_inputs(pi0_bundle)
    run_op_test(
        FullChain01_SecondEmbedSuffix(core),
        args,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_full_chain_second_02_rank3_mask_concat(request, pi0_bundle):
    core, args = _bundle_inputs(pi0_bundle)
    run_op_test(
        FullChain02_SecondThroughRank3Cat(core),
        args,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_full_chain_second_03_position_ids(request, pi0_bundle):
    core, args = _bundle_inputs(pi0_bundle)
    run_op_test(
        FullChain03_SecondThroughPositionIds(core),
        args,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_full_chain_second_04_attn_mask_4d(request, pi0_bundle):
    core, args = _bundle_inputs(pi0_bundle)
    run_op_test(
        FullChain04_SecondThroughAttnMask4d(core),
        args,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_full_chain_second_05_full_second_denoise(request, pi0_bundle):
    core, args = _bundle_inputs(pi0_bundle)
    run_op_test(
        FullChain05_SecondFullDenoiseStep(core),
        args,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_full_chain_baseline_two_step_sample_actions(request, pi0_bundle):
    """Same graph class as ``test_pi0_sample_actions_incremental_block_815_884_two_steps``."""
    core, args = _bundle_inputs(pi0_bundle)
    run_op_test(
        Pi0SampleActionsBlock815to884NumSteps(core, num_steps=2),
        args,
        framework=Framework.TORCH,
        request=request,
    )
