# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end TT bisect for **two denoise steps** in a **single traced forward**.

Relationship to ``test_pi0_second_denoise_bisect.py``:

- That file feeds **CPU-computed** ``x_t`` after denoise 1 into the second-denoise
  slice. All five tests can pass on TT while ``num_steps == 2`` full
  ``sample_actions`` still fails, because the failure may depend on **TT-produced**
  activations after the first denoise, **compiler fusion** across both steps, or
  graph size—not on the second-step subgraph alone with a frozen CPU midpoint.

- Here, **every** module runs **prefix KV + first ``denoise_step`` + (optional)
  prefix of the second ``denoise_step``** in one ``forward``, with inputs only
  ``[images, img_masks, lang_tokens, lang_masks, state, noise]``. The first failing
  test (run ``00`` → ``06``) is the **smallest included second-denoise boundary**
  that still reproduces the TT issue under a graph that includes a real first denoise
  on device.

``06`` matches the final ``x_t`` of ``sample_actions`` with ``num_steps == 2``
(same as ``Pi0SampleActionsBlock815to884NumSteps(..., 2)``).
"""

from __future__ import annotations

import pytest
import torch

from infra import Framework, run_op_test
from tests.torch.ops.pi0_pipeline_shared import (
    pi0_torch_cumsum_patch_like_model_py_forward,
    sample_actions_lines_815_884,
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


def _prefix_and_first_denoise(
    m: torch.nn.Module,
    images,
    img_masks,
    lang_tokens,
    lang_masks,
    state,
    noise: torch.Tensor,
    num_steps: int = 2,
):
    """``num_steps`` schedule with ``dt = -1/num_steps``; after first update."""
    bsize = state.shape[0]
    device = state.device
    dt = -1.0 / num_steps
    prefix_pad_masks, past_key_values = _prefix_pad_masks_and_pkv(
        m, images, img_masks, lang_tokens, lang_masks
    )
    x_t = noise
    time0 = torch.tensor(1.0 + 0 * dt, dtype=torch.float32, device=device).expand(
        bsize
    )
    _set_gemma_expert_attn_eager(m)
    v0 = m.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time0)
    x_t = x_t + dt * v0
    time1 = torch.tensor(1.0 + 1 * dt, dtype=torch.float32, device=device).expand(
        bsize
    )
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


# --- Single forward: prefix + denoise 1 on TT, then growing second-denoise prefix ---


class TwoDenoiseFullGraph00_AfterFirstDenoiseOnly(torch.nn.Module):
    """Prefix + first ``denoise_step`` only (``x_t`` after first Euler update)."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            _ppm, _pkv, x_t, _t1 = _prefix_and_first_denoise(
                m,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise,
                num_steps=self._num_steps,
            )
        return x_t


class TwoDenoiseFullGraph01_SecondEmbedSuffix(torch.nn.Module):
    """Through second ``embed_suffix`` only."""

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
            embs, _p, _a, _ad = m.embed_suffix(state, x_t, time1)
        return embs


class TwoDenoiseFullGraph02_SecondThroughRank3Cat(torch.nn.Module):
    """Through second-iteration ``torch.cat`` on rank-3 attention masks."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        from lerobot.policies.pi0 import modeling_pi0

        make_att_2d_masks = modeling_pi0.make_att_2d_masks
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            prefix_pad_masks, _pkv, x_t, time1 = _prefix_and_first_denoise(
                m,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise,
                num_steps=self._num_steps,
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


class TwoDenoiseFullGraph03_SecondThroughPositionIds(torch.nn.Module):
    """Through ``position_ids`` after rank-3 cat."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        from lerobot.policies.pi0 import modeling_pi0

        make_att_2d_masks = modeling_pi0.make_att_2d_masks
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            prefix_pad_masks, _pkv, x_t, time1 = _prefix_and_first_denoise(
                m,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise,
                num_steps=self._num_steps,
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
            full_att_2d_masks = torch.cat(
                [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
            )
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        return position_ids.to(dtype=torch.float32) + 1e-9 * full_att_2d_masks.to(
            dtype=torch.float32
        ).mean()


class TwoDenoiseFullGraph04_SecondThroughAttnMask4d(torch.nn.Module):
    """Through ``_prepare_attention_masks_4d`` on the second denoise."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        from lerobot.policies.pi0 import modeling_pi0

        make_att_2d_masks = modeling_pi0.make_att_2d_masks
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            prefix_pad_masks, _pkv, x_t, time1 = _prefix_and_first_denoise(
                m,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise,
                num_steps=self._num_steps,
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
            full_att_2d_masks = torch.cat(
                [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
            )
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            full_att_2d_masks_4d = m._prepare_attention_masks_4d(full_att_2d_masks)
        return full_att_2d_masks_4d.to(dtype=torch.float32) + 1e-9 * position_ids.to(
            dtype=torch.float32
        ).mean()


class TwoDenoiseFullGraph05_SecondFullDenoiseStep(torch.nn.Module):
    """Prefix + denoise 1 + full second ``denoise_step`` (velocity ``v_t`` for step 1)."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            prefix_pad_masks, past_key_values, x_t, time1 = _prefix_and_first_denoise(
                m,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise,
                num_steps=self._num_steps,
            )
            _set_gemma_expert_attn_eager(m)
            return m.denoise_step(
                state, prefix_pad_masks, past_key_values, x_t, time1
            )


class TwoDenoiseFullGraph06_FinalXTTwoSteps(torch.nn.Module):
    """Same final ``x_t`` as ``sample_actions`` with ``num_steps == 2``."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        return sample_actions_lines_815_884(
            self.core,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise,
            num_steps=self._num_steps,
            kwargs={},
        )


@pytest.mark.parametrize(
    "tag,mod_cls",
    [
        # ("00_after_first_denoise", TwoDenoiseFullGraph00_AfterFirstDenoiseOnly),
        ("01_second_embed_suffix", TwoDenoiseFullGraph01_SecondEmbedSuffix),
        # ("02_second_rank3_cat", TwoDenoiseFullGraph02_SecondThroughRank3Cat),
        # ("03_second_position_ids", TwoDenoiseFullGraph03_SecondThroughPositionIds),
        # ("04_second_attn_mask_4d", TwoDenoiseFullGraph04_SecondThroughAttnMask4d),
        # ("05_second_full_denoise_step", TwoDenoiseFullGraph05_SecondFullDenoiseStep),
        # ("06_final_x_t_two_steps", TwoDenoiseFullGraph06_FinalXTTwoSteps),
    ],
)
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_two_denoise_tt_fullgraph_bisect(request, pi0_bundle, tag, mod_cls):
    core, inputs = _bundle_inputs(pi0_bundle)
    run_op_test(
        mod_cls(core, num_steps=2),
        inputs,
        framework=Framework.TORCH,
        request=request,
    )
