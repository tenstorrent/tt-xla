# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Finer bisect inside **second-step** ``embed_suffix`` on TT, after a real first denoise.

``tt_denoise_steps_11.log`` shows ``test_pi0_two_denoise_tt_fullgraph_bisect[01_second_embed_suffix]``
failing on ``ttnn::concat`` while ``00_after_first_denoise`` passes—so the reproducer
includes ``embed_suffix(state, x_t, time1)`` but not necessarily the whole method as an
opaque call. This file mirrors ``PI0Pytorch.embed_suffix`` (``modeling_pi0`` ~680--741)
in **cumulative** slices: each module runs ``_prefix_and_first_denoise`` then executes
``embed_suffix`` only **up to** a named boundary.

Run tests in tag order ``s00`` → ``s07`` (with ``s02a``--``s02d`` between ``s01`` and ``s03``).
The **first** failure isolates the smallest
prefix of ``embed_suffix`` (after TT first denoise) that still triggers the TT error.

Sub-ladder when ``s02``-class failure is narrowed (``action_in_proj`` / ``nn.Linear``):

  ``s02a`` — ``x_t`` from the first denoise appears in the output graph only (no linear).
  ``s02b`` — ``F.linear(x_t, weight, bias=None)`` (matmul path, no bias).
  ``s02c`` — ``F.linear(x_t, weight, bias)`` (full linear, no checkpoint wrapper).
  ``s02d`` — same as upstream: ``_apply_checkpoint(action_in_proj, x_t)``.

Upstream op order (approximate):

  state_proj → time sinusoid → ``action_in_proj`` → ``torch.cat(..., dim=2)``
  (action/time) → ``action_time_mlp`` → ``torch.cat(embs, dim=1)``
  → ``torch.cat(pad_masks, dim=1)`` → ``att_masks`` tensor + expand.

``s07`` matches the full second ``embed_suffix`` embedding output (same subgraph as
``TwoDenoiseFullGraph01_SecondEmbedSuffix``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from infra import Framework, run_op_test
from tests.torch.ops.pi0_pipeline_shared import pi0_torch_cumsum_patch_like_model_py_forward
from tests.torch.ops.test_pi0_two_denoise_tt_fullgraph_bisect import _prefix_and_first_denoise
from utils import Category


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


def _state_for_suffix(m: torch.nn.Module, state: torch.Tensor) -> torch.Tensor:
    if m.state_proj.weight.dtype == torch.float32:
        return state.to(torch.float32)
    return state


def _create_sinusoidal_pos_embedding(
    timestep, out_features, *, min_period, max_period, device
):
    """Lazy import: avoid importing ``lerobot...modeling_pi0`` at collection time.

    A top-level import there can leave ``transformers.models.paligemma.modeling_paligemma``
    out of ``sys.modules`` and break ``PI0Policy.from_pretrained`` with
    ``KeyError`` in Transformers ``_can_set_experts_implementation`` (``tt_embed_suffix.log``).
    """
    from lerobot.policies.pi0 import modeling_pi0

    return modeling_pi0.create_sinusoidal_pos_embedding(
        timestep,
        out_features,
        min_period=min_period,
        max_period=max_period,
        device=device,
    )


class SecondEmbedSuffixTT_S00_StateProj(torch.nn.Module):
    """After first denoise: ``state_proj`` + ``state_emb[:, None, :]`` only."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            _ppm, _pkv, _x_t, _time1 = _prefix_and_first_denoise(
                m,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise,
                num_steps=self._num_steps,
            )
            state_in = _state_for_suffix(m, state)

            def state_proj_func(s):
                return m.state_proj(s)

            state_emb = m._apply_checkpoint(state_proj_func, state_in)
            state_token = state_emb[:, None, :]
        return state_token


class SecondEmbedSuffixTT_S01_PlusTimeSinusoid(torch.nn.Module):
    """+ ``create_sinusoidal_pos_embedding`` / dtype cast."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            _ppm, _pkv, x_t, time1 = _prefix_and_first_denoise(
                m,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise,
                num_steps=self._num_steps,
            )
            state_in = _state_for_suffix(m, state)

            def state_proj_func(s):
                return m.state_proj(s)

            state_emb = m._apply_checkpoint(state_proj_func, state_in)
            state_token = state_emb[:, None, :]

            time_emb = _create_sinusoidal_pos_embedding(
                time1,
                m.action_in_proj.out_features,
                min_period=m.config.min_period,
                max_period=m.config.max_period,
                device=time1.device,
            )
            time_emb = time_emb.type(dtype=time1.dtype)
        return time_emb + 1e-9 * state_token.mean()


class SecondEmbedSuffixTT_S02a_XtInOutputOnly(torch.nn.Module):
    """After time path: ``x_t`` in graph via output mix only (no ``action_in_proj``)."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            _ppm, _pkv, x_t, time1 = _prefix_and_first_denoise(
                m,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise,
                num_steps=self._num_steps,
            )
            state_in = _state_for_suffix(m, state)

            def state_proj_func(s):
                return m.state_proj(s)

            state_emb = m._apply_checkpoint(state_proj_func, state_in)
            state_token = state_emb[:, None, :]

            time_emb = _create_sinusoidal_pos_embedding(
                time1,
                m.action_in_proj.out_features,
                min_period=m.config.min_period,
                max_period=m.config.max_period,
                device=time1.device,
            )
            time_emb = time_emb.type(dtype=time1.dtype)
        return time_emb + 1e-9 * (state_token.mean() + x_t.mean())


class SecondEmbedSuffixTT_S02b_FLinearNoBias(torch.nn.Module):
    """+ ``F.linear(x_t, action_in_proj.weight, bias=None)`` (matmul, no bias)."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            _ppm, _pkv, x_t, time1 = _prefix_and_first_denoise(
                m,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise,
                num_steps=self._num_steps,
            )
            state_in = _state_for_suffix(m, state)

            def state_proj_func(s):
                return m.state_proj(s)

            state_emb = m._apply_checkpoint(state_proj_func, state_in)
            state_token = state_emb[:, None, :]

            time_emb = _create_sinusoidal_pos_embedding(
                time1,
                m.action_in_proj.out_features,
                min_period=m.config.min_period,
                max_period=m.config.max_period,
                device=time1.device,
            )
            time_emb = time_emb.type(dtype=time1.dtype)

            lin = m.action_in_proj
            action_core = F.linear(x_t, lin.weight, None)
        return action_core + 1e-9 * (state_token.mean() + time_emb.mean())


class SecondEmbedSuffixTT_S02c_FLinearWithBias(torch.nn.Module):
    """+ ``F.linear(x_t, weight, bias)`` (full ``nn.Linear`` math, no checkpoint)."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            _ppm, _pkv, x_t, time1 = _prefix_and_first_denoise(
                m,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise,
                num_steps=self._num_steps,
            )
            state_in = _state_for_suffix(m, state)

            def state_proj_func(s):
                return m.state_proj(s)

            state_emb = m._apply_checkpoint(state_proj_func, state_in)
            state_token = state_emb[:, None, :]

            time_emb = _create_sinusoidal_pos_embedding(
                time1,
                m.action_in_proj.out_features,
                min_period=m.config.min_period,
                max_period=m.config.max_period,
                device=time1.device,
            )
            time_emb = time_emb.type(dtype=time1.dtype)

            lin = m.action_in_proj
            action_full = F.linear(x_t, lin.weight, lin.bias)
        return action_full + 1e-9 * (state_token.mean() + time_emb.mean())


class SecondEmbedSuffixTT_S02d_ActionInProjCheckpoint(torch.nn.Module):
    """+ ``_apply_checkpoint(action_in_proj, x_t)`` (matches ``embed_suffix``)."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            _ppm, _pkv, x_t, time1 = _prefix_and_first_denoise(
                m,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise,
                num_steps=self._num_steps,
            )
            state_in = _state_for_suffix(m, state)

            def state_proj_func(s):
                return m.state_proj(s)

            state_emb = m._apply_checkpoint(state_proj_func, state_in)
            state_token = state_emb[:, None, :]

            time_emb = _create_sinusoidal_pos_embedding(
                time1,
                m.action_in_proj.out_features,
                min_period=m.config.min_period,
                max_period=m.config.max_period,
                device=time1.device,
            )
            time_emb = time_emb.type(dtype=time1.dtype)

            def action_proj_func(noisy_actions):
                return m.action_in_proj(noisy_actions)

            action_emb = m._apply_checkpoint(action_proj_func, x_t)
        return action_emb + 1e-9 * (state_token.mean() + time_emb.mean())


class SecondEmbedSuffixTT_S03_PlusCatActionTimeDim2(torch.nn.Module):
    """+ ``torch.cat([action_emb, time_emb], dim=2)``."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            _ppm, _pkv, x_t, time1 = _prefix_and_first_denoise(
                m,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise,
                num_steps=self._num_steps,
            )
            state_in = _state_for_suffix(m, state)

            def state_proj_func(s):
                return m.state_proj(s)

            state_emb = m._apply_checkpoint(state_proj_func, state_in)
            state_token = state_emb[:, None, :]

            time_emb = _create_sinusoidal_pos_embedding(
                time1,
                m.action_in_proj.out_features,
                min_period=m.config.min_period,
                max_period=m.config.max_period,
                device=time1.device,
            )
            time_emb = time_emb.type(dtype=time1.dtype)

            def action_proj_func(noisy_actions):
                return m.action_in_proj(noisy_actions)

            action_emb = m._apply_checkpoint(action_proj_func, x_t)
            time_exp = time_emb[:, None, :].expand_as(action_emb)
            action_time_cat = torch.cat([action_emb, time_exp], dim=2)
        return action_time_cat + 1e-9 * state_token.mean()


class SecondEmbedSuffixTT_S04_PlusActionTimeMlp(torch.nn.Module):
    """+ ``action_time_mlp_in`` / silu / ``action_time_mlp_out``."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            _ppm, _pkv, x_t, time1 = _prefix_and_first_denoise(
                m,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise,
                num_steps=self._num_steps,
            )
            state_in = _state_for_suffix(m, state)

            def state_proj_func(s):
                return m.state_proj(s)

            state_emb = m._apply_checkpoint(state_proj_func, state_in)
            state_token = state_emb[:, None, :]

            time_emb = _create_sinusoidal_pos_embedding(
                time1,
                m.action_in_proj.out_features,
                min_period=m.config.min_period,
                max_period=m.config.max_period,
                device=time1.device,
            )
            time_emb = time_emb.type(dtype=time1.dtype)

            def action_proj_func(noisy_actions):
                return m.action_in_proj(noisy_actions)

            action_emb = m._apply_checkpoint(action_proj_func, x_t)
            time_exp = time_emb[:, None, :].expand_as(action_emb)
            action_time_cat = torch.cat([action_emb, time_exp], dim=2)

            def mlp_func(action_time_emb):
                x = m.action_time_mlp_in(action_time_emb)
                x = F.silu(x)
                return m.action_time_mlp_out(x)

            action_time_emb = m._apply_checkpoint(mlp_func, action_time_cat)
        return action_time_emb + 1e-9 * state_token.mean()


class SecondEmbedSuffixTT_S05_PlusCatEmbsDim1(torch.nn.Module):
    """+ ``torch.cat(embs, dim=1)`` (state token + action/time sequence)."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            _ppm, _pkv, x_t, time1 = _prefix_and_first_denoise(
                m,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise,
                num_steps=self._num_steps,
            )
            state_in = _state_for_suffix(m, state)

            def state_proj_func(s):
                return m.state_proj(s)

            state_emb = m._apply_checkpoint(state_proj_func, state_in)
            state_token = state_emb[:, None, :]
            bsize = state_emb.shape[0]
            device = state_emb.device

            time_emb = _create_sinusoidal_pos_embedding(
                time1,
                m.action_in_proj.out_features,
                min_period=m.config.min_period,
                max_period=m.config.max_period,
                device=time1.device,
            )
            time_emb = time_emb.type(dtype=time1.dtype)

            def action_proj_func(noisy_actions):
                return m.action_in_proj(noisy_actions)

            action_emb = m._apply_checkpoint(action_proj_func, x_t)
            time_exp = time_emb[:, None, :].expand_as(action_emb)
            action_time_cat = torch.cat([action_emb, time_exp], dim=2)

            def mlp_func(action_time_emb):
                x = m.action_time_mlp_in(action_time_emb)
                x = F.silu(x)
                return m.action_time_mlp_out(x)

            action_time_emb = m._apply_checkpoint(mlp_func, action_time_cat)

            embs = [state_token, action_time_emb]
            seq_embs = torch.cat(embs, dim=1)
        return seq_embs


class SecondEmbedSuffixTT_S06_PlusCatPadMasksDim1(torch.nn.Module):
    """+ ``torch.cat(pad_masks, dim=1)``."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            _ppm, _pkv, x_t, time1 = _prefix_and_first_denoise(
                m,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise,
                num_steps=self._num_steps,
            )
            state_in = _state_for_suffix(m, state)

            def state_proj_func(s):
                return m.state_proj(s)

            state_emb = m._apply_checkpoint(state_proj_func, state_in)
            state_token = state_emb[:, None, :]
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks = [state_mask]

            time_emb = _create_sinusoidal_pos_embedding(
                time1,
                m.action_in_proj.out_features,
                min_period=m.config.min_period,
                max_period=m.config.max_period,
                device=time1.device,
            )
            time_emb = time_emb.type(dtype=time1.dtype)

            def action_proj_func(noisy_actions):
                return m.action_in_proj(noisy_actions)

            action_emb = m._apply_checkpoint(action_proj_func, x_t)
            time_exp = time_emb[:, None, :].expand_as(action_emb)
            action_time_cat = torch.cat([action_emb, time_exp], dim=2)

            def mlp_func(action_time_emb):
                x = m.action_time_mlp_in(action_time_emb)
                x = F.silu(x)
                return m.action_time_mlp_out(x)

            action_time_emb = m._apply_checkpoint(mlp_func, action_time_cat)
            _, action_time_dim = action_time_emb.shape[:2]
            action_time_mask = torch.ones(
                bsize, action_time_dim, dtype=torch.bool, device=time1.device
            )
            pad_masks.append(action_time_mask)

            embs = [state_token, action_time_emb]
            seq_embs = torch.cat(embs, dim=1)
            pad_cat = torch.cat(pad_masks, dim=1)
        return pad_cat.to(dtype=torch.float32) + 1e-9 * seq_embs.mean()


class SecondEmbedSuffixTT_S07_FullEmbedSuffixEmbs(torch.nn.Module):
    """Full ``embed_suffix`` → return suffix embeddings (matches fullgraph ``01``)."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            _ppm, _pkv, x_t, time1 = _prefix_and_first_denoise(
                m,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise,
                num_steps=self._num_steps,
            )
            embs, _pad, _att, _ada = m.embed_suffix(state, x_t, time1)
        return embs


@pytest.mark.parametrize(
    "tag,mod_cls",
    [
        ("s00_state_proj", SecondEmbedSuffixTT_S00_StateProj),
        ("s01_plus_time_sinusoid", SecondEmbedSuffixTT_S01_PlusTimeSinusoid),
        ("s02a_xt_in_output_only", SecondEmbedSuffixTT_S02a_XtInOutputOnly),
        ("s02b_f_linear_no_bias", SecondEmbedSuffixTT_S02b_FLinearNoBias),
        ("s02c_f_linear_with_bias", SecondEmbedSuffixTT_S02c_FLinearWithBias),
        ("s02d_action_in_proj_checkpoint", SecondEmbedSuffixTT_S02d_ActionInProjCheckpoint),
        ("s03_plus_cat_action_time_dim2", SecondEmbedSuffixTT_S03_PlusCatActionTimeDim2),
        ("s04_plus_action_time_mlp", SecondEmbedSuffixTT_S04_PlusActionTimeMlp),
        ("s05_plus_cat_embs_dim1", SecondEmbedSuffixTT_S05_PlusCatEmbsDim1),
        ("s06_plus_cat_pad_masks_dim1", SecondEmbedSuffixTT_S06_PlusCatPadMasksDim1),
        ("s07_full_embed_suffix_embs", SecondEmbedSuffixTT_S07_FullEmbedSuffixEmbs),
    ],
)
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_second_embed_suffix_incremental_tt(request, pi0_bundle, tag, mod_cls):
    core, inputs = _bundle_inputs(pi0_bundle)
    run_op_test(
        mod_cls(core, num_steps=2),
        inputs,
        framework=Framework.TORCH,
        request=request,
    )
