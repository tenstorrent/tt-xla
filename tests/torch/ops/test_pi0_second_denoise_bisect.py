# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Bisect **second** ``PI0Pytorch.denoise_step`` (same state as ``num_steps == 2`` after
step 0) to see which front-end op first triggers TT failures (e.g. ``ttnn::concat`` /
``ShapeBase[] index out of range``).

``num_steps == 1`` runs the denoise body once; ``num_steps == 2`` is the smallest
loop that exercises the same ``denoise_step`` subgraph twice on one graph, which is
why it is the usual minimal sanity beyond a single step.

Flow:

1. On CPU (no TT), run prefix KV + **one** denoise iteration with ``num_steps=2``
   (``dt = -0.5``, first time at ``t=1.0``) to get ``x_t`` after the first update.
2. ``run_op_test`` modules that each recompute prefix KV from the bundle inputs (so
   device placement matches other Pi0 op tests), then run an **increasing prefix** of
   the **second** denoise (``t = 0.5``) and return a single float tensor for PCC.

Interpretation: run tests in order ``01`` → ``05``. The **first** failing test pins
the earliest boundary in ``denoise_step`` where TT breaks for that second-iteration
state. Upstream order in ``modeling_pi0``::

    embed_suffix → mask expand / make_att_2d_masks → torch.cat(dim=2)
    → position_ids → _prepare_attention_masks_4d → paligemma_with_expert.forward
    → chunk / dtype / action_out_proj
"""

from __future__ import annotations

import pytest
import torch

from infra import Framework, run_op_test
from tests.torch.ops.pi0_pipeline_shared import (
    pi0_torch_cumsum_patch_like_model_py_forward,
    set_paligemma_lm_attn_eager,
)
from utils import Category


def _set_gemma_expert_attn_eager(m: torch.nn.Module) -> None:
    """Match ``PI0Pytorch.denoise_step`` (suffix expert path)."""
    exp = m.paligemma_with_expert.gemma_expert
    exp.model.config._attn_implementation = "eager"  # noqa: SLF001


def _prefix_pad_masks_and_pkv(
    m: torch.nn.Module,
    images,
    img_masks,
    lang_tokens,
    lang_masks,
):
    """Prefix KV; caller must wrap with ``pi0_torch_cumsum_patch_like_model_py_forward`` when needed."""
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


def cpu_x_t_and_time_after_first_denoise(
    m: torch.nn.Module,
    images,
    img_masks,
    lang_tokens,
    lang_masks,
    state,
    noise: torch.Tensor,
    num_steps: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefix + first denoise only; returns ``(x_t_after_step0, time_tensor_step1)``."""
    bsize = state.shape[0]
    device = state.device
    dt = -1.0 / num_steps
    with pi0_torch_cumsum_patch_like_model_py_forward():
        with torch.no_grad():
            prefix_pad_masks, past_key_values = _prefix_pad_masks_and_pkv(
                m, images, img_masks, lang_tokens, lang_masks
            )
            x_t = noise
            time0 = torch.tensor(1.0 + 0 * dt, dtype=torch.float32, device=device).expand(
                bsize
            )
            _set_gemma_expert_attn_eager(m)
            v0 = m.denoise_step(
                state, prefix_pad_masks, past_key_values, x_t, time0
            )
            x_t = x_t + dt * v0
            time1 = torch.tensor(1.0 + 1 * dt, dtype=torch.float32, device=device).expand(
                bsize
            )
    return x_t, time1


def _second_denoise_run_op_inputs(
    core,
    images,
    img_masks,
    lang_tokens,
    lang_masks,
    state,
    noise,
    num_steps: int = 2,
):
    x_t_mid, time1 = cpu_x_t_and_time_after_first_denoise(
        core, images, img_masks, lang_tokens, lang_masks, state, noise, num_steps=num_steps
    )
    return [images, img_masks, lang_tokens, lang_masks, state, noise, x_t_mid, time1]


# --- Incremental second-iteration slices (each recomputes prefix KV inside forward) ---


class SecondDenoise01_EmbedSuffixOnly(torch.nn.Module):
    """Second denoise: ``embed_suffix`` only."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
        x_t_mid,
        time_step1,
    ):
        del images, img_masks, lang_tokens, lang_masks, noise
        m = self.core
        suffix_embs, _pad, _att, _ada = m.embed_suffix(state, x_t_mid, time_step1)
        return suffix_embs


class SecondDenoise02_ThroughRank3Cat(torch.nn.Module):
    """Through ``torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)``."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
        x_t_mid,
        time_step1,
    ):
        from lerobot.policies.pi0 import modeling_pi0

        make_att_2d_masks = modeling_pi0.make_att_2d_masks
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            prefix_pad_masks, _pkv = _prefix_pad_masks_and_pkv(
                m, images, img_masks, lang_tokens, lang_masks
            )
            _embs, suffix_pad_masks, suffix_att_masks, _ada = m.embed_suffix(
                state, x_t_mid, time_step1
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


class SecondDenoise03_PlusPositionIds(torch.nn.Module):
    """Through ``position_ids`` (after rank-3 cat)."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
        x_t_mid,
        time_step1,
    ):
        from lerobot.policies.pi0 import modeling_pi0

        make_att_2d_masks = modeling_pi0.make_att_2d_masks
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            prefix_pad_masks, _pkv = _prefix_pad_masks_and_pkv(
                m, images, img_masks, lang_tokens, lang_masks
            )
            _embs, suffix_pad_masks, suffix_att_masks, _ada = m.embed_suffix(
                state, x_t_mid, time_step1
            )
            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat(
                [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
            )
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        # Keep rank-3 concat in the graph (avoid DCE of ``full_att_2d_masks``).
        return position_ids.to(dtype=torch.float32) + 1e-9 * full_att_2d_masks.to(
            dtype=torch.float32
        ).mean()


class SecondDenoise04_PlusAttnMask4d(torch.nn.Module):
    """Through ``_prepare_attention_masks_4d``."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
        x_t_mid,
        time_step1,
    ):
        from lerobot.policies.pi0 import modeling_pi0

        make_att_2d_masks = modeling_pi0.make_att_2d_masks
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            prefix_pad_masks, _pkv = _prefix_pad_masks_and_pkv(
                m, images, img_masks, lang_tokens, lang_masks
            )
            _embs, suffix_pad_masks, suffix_att_masks, _ada = m.embed_suffix(
                state, x_t_mid, time_step1
            )
            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len
            )
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat(
                [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
            )
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            full_att_2d_masks_4d = m._prepare_attention_masks_4d(full_att_2d_masks)
        # Keep ``position_ids`` in the graph before ``_prepare_attention_masks_4d``.
        return full_att_2d_masks_4d.to(dtype=torch.float32) + 1e-9 * position_ids.to(
            dtype=torch.float32
        ).mean()


class SecondDenoise05_FullStep(torch.nn.Module):
    """Full ``denoise_step`` for the second iteration (same as one ``denoise_step`` call)."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
        x_t_mid,
        time_step1,
    ):
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            prefix_pad_masks, past_key_values = _prefix_pad_masks_and_pkv(
                m, images, img_masks, lang_tokens, lang_masks
            )
            _set_gemma_expert_attn_eager(m)
            return m.denoise_step(
                state, prefix_pad_masks, past_key_values, x_t_mid, time_step1
            )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_second_denoise_bisect_01_embed_suffix(request, pi0_bundle):
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    run_op_test(
        SecondDenoise01_EmbedSuffixOnly(core),
        _second_denoise_run_op_inputs(
            core, images, img_masks, lang_tokens, lang_masks, state, noise
        ),
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_second_denoise_bisect_02_rank3_mask_concat(request, pi0_bundle):
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    run_op_test(
        SecondDenoise02_ThroughRank3Cat(core),
        _second_denoise_run_op_inputs(
            core, images, img_masks, lang_tokens, lang_masks, state, noise
        ),
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_second_denoise_bisect_03_position_ids(request, pi0_bundle):
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    run_op_test(
        SecondDenoise03_PlusPositionIds(core),
        _second_denoise_run_op_inputs(
            core, images, img_masks, lang_tokens, lang_masks, state, noise
        ),
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_second_denoise_bisect_04_attn_mask_4d(request, pi0_bundle):
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    run_op_test(
        SecondDenoise04_PlusAttnMask4d(core),
        _second_denoise_run_op_inputs(
            core, images, img_masks, lang_tokens, lang_masks, state, noise
        ),
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_second_denoise_bisect_05_full_denoise_step(request, pi0_bundle):
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    run_op_test(
        SecondDenoise05_FullStep(core),
        _second_denoise_run_op_inputs(
            core, images, img_masks, lang_tokens, lang_masks, state, noise
        ),
        framework=Framework.TORCH,
        request=request,
    )
