# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared Pi0 ``sample_actions`` helpers and wrappers for ``tests/torch/ops/``."""

from __future__ import annotations

import contextlib

import torch


@contextlib.contextmanager
def pi0_torch_cumsum_patch_like_model_py_forward():
    """
    Same monkey-patch as ``pi_0/pytorch/src/model.py`` ``forward`` (lines 96--109):
    ``torch.cumsum`` on ``bool`` inputs casts to ``torch.long`` first.
    """
    original_cumsum = torch.cumsum

    def _safe_cumsum(input, dim, **kwargs):
        if input.dtype == torch.bool:
            input = input.to(torch.long)
        return original_cumsum(input, dim, **kwargs)

    torch.cumsum = _safe_cumsum
    try:
        yield
    finally:
        torch.cumsum = original_cumsum


def crop_actions_chunk_like_model_py(policy: torch.nn.Module, actions: torch.Tensor) -> torch.Tensor:
    """``pi_0/pytorch/src/model.py`` ``forward`` lines 111--113 (after ``sample_actions``, before queue)."""
    original_action_dim = policy.config.output_features["action"].shape[0]
    actions = actions[:, :, :original_action_dim]
    actions = actions[:, : policy.config.n_action_steps]
    return actions


def clear_policy_device_queue(policy: torch.nn.Module, device: torch.device) -> None:
    """Drop cached queue for ``device`` so the next ``forward`` runs ``sample_actions`` again."""
    queues = getattr(policy, "_device_queues", None)
    if isinstance(queues, dict):
        queues.pop(str(device), None)


def set_paligemma_lm_attn_eager(paligemma: torch.nn.Module) -> None:
    """Match ``modeling_pi0.sample_actions`` (layout differs across HF/lerobot builds)."""
    if hasattr(paligemma, "language_model"):
        paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    elif hasattr(paligemma, "model") and hasattr(paligemma.model, "language_model"):
        paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    else:
        raise AttributeError("Cannot find paligemma.language_model for attn override")


def prefix_tensors_before_paligemma_forward(
    core: torch.nn.Module,
    images,
    img_masks,
    lang_tokens,
    lang_masks,
):
    """Shared prefix path with ``PI0Pytorch.sample_actions`` (before ``paligemma_with_expert.forward``)."""
    from lerobot.policies.pi0 import modeling_pi0

    m = core
    with pi0_torch_cumsum_patch_like_model_py_forward():
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = modeling_pi0.make_att_2d_masks(
            prefix_pad_masks, prefix_att_masks
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = m._prepare_attention_masks_4d(prefix_att_2d_masks)
    return prefix_embs, prefix_att_2d_masks_4d, prefix_position_ids


def sample_actions_lines_815_845(
    m: torch.nn.Module,
    images,
    img_masks,
    lang_tokens,
    lang_masks,
    state,
    noise: torch.Tensor | None,
) -> torch.Tensor:
    """Mirrors ``PI0Pytorch.sample_actions`` through prefix KV (~815--845)."""
    from lerobot.policies.pi0 import modeling_pi0

    make_att_2d_masks = modeling_pi0.make_att_2d_masks

    with pi0_torch_cumsum_patch_like_model_py_forward():
        num_steps = None
        if num_steps is None:
            num_steps = m.config.num_inference_steps
        _ = num_steps

        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (
                bsize,
                m.config.chunk_size,
                m.config.max_action_dim,
            )
            noise = m.sample_noise(actions_shape, device)

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
        return past_key_values


def sample_actions_lines_815_884(
    m: torch.nn.Module,
    images,
    img_masks,
    lang_tokens,
    lang_masks,
    state,
    noise: torch.Tensor | None,
    num_steps: int | None,
    kwargs: dict | None = None,
) -> torch.Tensor:
    """Full ``sample_actions`` body through the denoise loop (~815--884)."""
    from lerobot.policies.pi0 import modeling_pi0

    make_att_2d_masks = modeling_pi0.make_att_2d_masks
    if kwargs is None:
        kwargs = {}

    if num_steps is None:
        num_steps = m.config.num_inference_steps
    num_steps = int(num_steps)

    with pi0_torch_cumsum_patch_like_model_py_forward():
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (
                bsize,
                m.config.chunk_size,
                m.config.max_action_dim,
            )
            noise = m.sample_noise(actions_shape, device)

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

        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(
                bsize
            )

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return m.denoise_step(
                    state=state,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if m._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = m.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if m.rtc_processor is not None and m.rtc_processor.is_debug_enabled():
                m.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t


class Pi0SampleActionsBlock815to845(torch.nn.Module):
    """``sample_actions`` lines ~815--845 only: noise setup, prefix, ``past_key_values``."""

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
        noise: torch.Tensor | None,
    ):
        return sample_actions_lines_815_845(
            self.core, images, img_masks, lang_tokens, lang_masks, state, noise
        )


class Pi0SampleActionsBlock815to884NumSteps(torch.nn.Module):
    """``sample_actions`` through line ~884; ``num_steps`` ``None`` uses ``config``."""

    def __init__(self, core: torch.nn.Module, num_steps: int | None):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise: torch.Tensor | None,
    ):
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


class Pi0IncrementalFullStepsPlusPolicyCrop(torch.nn.Module):
    """Full incremental ``sample_actions`` (``num_steps=None``) + ``model.py`` crop (no queue)."""

    def __init__(self, core: torch.nn.Module, policy: torch.nn.Module):
        super().__init__()
        self.add_module("_core", core)
        self.add_module("_policy", policy)

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise: torch.Tensor | None,
    ):
        x_t = sample_actions_lines_815_884(
            self._core,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise,
            num_steps=None,
            kwargs={},
        )
        return crop_actions_chunk_like_model_py(self._policy, x_t)


class Pi0PolicyForwardLikeCustomModelPy(torch.nn.Module):
    """
    Invokes ``PI0Policy.forward`` from ``pi_0/pytorch/src/model.py`` like the forge runner.

    Clears the per-device action queue before each call so CPU/TT runs both execute a
    full ``sample_actions`` + crop + first ``popleft`` path.
    """

    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.add_module("_policy", policy)

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise: torch.Tensor | None,
    ):
        clear_policy_device_queue(self._policy, state.device)
        return self._policy.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
        )
