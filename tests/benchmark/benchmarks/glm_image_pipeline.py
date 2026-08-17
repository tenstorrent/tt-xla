# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-Image — benchmark-side pipeline for the imagegen harness.

GLM-Image is a diffusion text-to-image model whose DiT transformer runs
**tensor-parallel** across a multi-chip mesh, while the AR vision-language
encoder, the T5 glyph text encoder, the FlowMatchEuler scheduler and the VAE
decode stay on CPU. The end-to-end pipeline (device split + DiT sharding) lives
in ``tt_forge_models`` and is the same one the nightly pipeline test drives.

The imagegen harness (``benchmarks/imagegen_benchmark.py``) needs two things the
base ``GlmImagePipeline`` does not provide:

  - a ``generate(prompt, num_inference_steps, ...)`` that accepts the step count
    per call (warmup uses 1 step, steady-state uses the full count), and
  - a ``self._perf`` dict populated with per-component / per-step timings.

So this module subclasses the ``tt_forge_models`` pipeline and overrides
``generate`` to add both, without duplicating the model loading / sharding
setup. The denoising body mirrors the base pipeline's t2i path exactly; the only
additions are the ``num_inference_steps`` override and the timing hooks.
"""

import time
from typing import Optional

import numpy as np
import torch
import torch_xla.core.xla_model as xm
from loguru import logger

from third_party.tt_forge_models.glm_image.pytorch.src.pipeline import (
    CPU_DTYPE,
    HEIGHT,
    PROMPT,
    SEED,
    TRANSFORMER_DTYPE,
    WIDTH,
    GlmImageConfig,
)
from third_party.tt_forge_models.glm_image.pytorch.src.pipeline import (
    GlmImagePipeline as _BaseGlmImagePipeline,
)

__all__ = ["GlmImageConfig", "GlmImagePipeline", "PROMPT", "SEED", "HEIGHT", "WIDTH"]


class GlmImagePipeline(_BaseGlmImagePipeline):
    """GLM-Image pipeline with benchmark-harness timing instrumentation.

    Reuses the base pipeline's model loading and DiT tensor-parallel sharding
    (``setup``/``load_models``/``shard_to_tt``) and only overrides ``generate``
    to accept a per-call ``num_inference_steps`` and to populate ``self._perf``.
    """

    @torch.no_grad()
    def generate(
        self,
        prompt: str = PROMPT,
        seed: Optional[int] = SEED,
        num_inference_steps: Optional[int] = None,
        output_type: str = "latent",
    ):
        """Run the GLM-Image t2i pipeline and record per-stage timings.

        Mirrors ``tt_forge_models`` ``GlmImagePipeline.generate`` (CPU AR prior
        tokens + T5 encode + FlowMatchEuler scheduler + VAE decode; DiT denoise
        on TT), adding:

          - ``num_inference_steps``: overrides ``config.num_inference_steps`` for
            this call (the harness warms up with 1 step, then runs the full count).
          - ``self._perf``: harness-readable timings. ``components`` holds scalar
            per-stage seconds (AR prior tokens, T5 encode, VAE decode -- all CPU),
            ``steps`` holds per-DiT-step seconds (the TT-resident denoise work;
            the ``_to_cpu`` cast after each step's forwards forces a device sync
            so the timer captures real device time), ``step_metric_name`` labels
            the per-step metric, and ``total`` is the full ``generate`` wall time.

        Returns the raw VAE decode ``(B, 3, H, W)`` in ``[-1, 1]`` when
        ``output_type="latent"`` (what ``utils.save_image`` expects); other
        ``output_type`` values match the base pipeline's post-processing.
        """
        from diffusers.pipelines.glm_image.pipeline_glm_image import (
            calculate_shift,
            retrieve_timesteps,
        )

        self._perf = {
            "components": {},
            "steps": [],
            "step_metric_name": "transformer_step",
            "total": None,
        }
        t_total_start = time.perf_counter()

        pipe = self.pipe
        transformer = self.transformer
        vae = self.vae
        scheduler = self.scheduler
        on_tt = self.config.transformer_on_tt
        cpu = torch.device("cpu")

        height, width = self.config.height, self.config.width
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = self.config.guidance_scale
        do_cfg = guidance_scale > 1
        B = 1

        def _to_tt(x, dtype=None):
            if not on_tt:
                return x
            if dtype is not None:
                x = x.to(dtype)
            return x.to(xm.xla_device())

        def _to_cpu(x):
            return x.to("cpu") if on_tt else x

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        # ── AR prior-token generation (CPU, vision-language encoder) ──────
        logger.info("[STAGE] AR prior-token generation (CPU): start")
        t0 = time.perf_counter()
        prior_token_ids, _, _ = pipe.generate_prior_tokens(
            prompt=prompt,
            image=None,  # text-to-image
            height=height,
            width=width,
            device=cpu,
            generator=generator,
        )
        self._perf["components"]["ar_prior_tokens"] = time.perf_counter() - t0
        logger.info("[STAGE] AR prior-token generation (CPU): done")

        # ── T5 glyph text encode (CPU) ────────────────────────────────────
        logger.info("[STAGE] T5 glyph text encode (CPU): start")
        t0 = time.perf_counter()
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt,
            do_classifier_free_guidance=do_cfg,
            num_images_per_prompt=1,
            device=cpu,
            dtype=CPU_DTYPE,
            max_sequence_length=self.config.max_sequence_length,
        )
        self._perf["components"]["t5_encode"] = time.perf_counter() - t0
        logger.info("[STAGE] T5 glyph text encode (CPU): done")

        # ── Latents + timestep conditioning (CPU) ─────────────────────────
        latents = pipe.prepare_latents(
            batch_size=B,
            num_channels_latents=transformer.config.in_channels,
            height=height,
            width=width,
            dtype=CPU_DTYPE,
            device=cpu,
            generator=generator,
        )
        target_size = torch.tensor([[height, width]], dtype=CPU_DTYPE)
        crop_coords = torch.tensor([[0, 0]], dtype=CPU_DTYPE)

        # ── Timesteps (FlowMatchEuler with resolution-dependent shift) ─────
        image_seq_len = (
            (height // pipe.vae_scale_factor) * (width // pipe.vae_scale_factor)
        ) // (transformer.config.patch_size**2)
        timesteps = np.linspace(
            scheduler.config.num_train_timesteps, 1.0, num_inference_steps + 1
        )[:-1]
        timesteps = timesteps.astype(np.int64).astype(np.float32)
        sigmas = timesteps / scheduler.config.num_train_timesteps
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("base_shift", 0.25),
            scheduler.config.get("max_shift", 0.75),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, cpu, timesteps, sigmas, mu=mu
        )

        # ── Loop-invariant DiT inputs: cast to bf16 + move to TT once ──────
        prior_ids_tt = _to_tt(prior_token_ids)
        drop_cond_tt = _to_tt(torch.full_like(prior_token_ids, False, dtype=torch.bool))
        drop_uncond_tt = _to_tt(
            torch.full_like(prior_token_ids, True, dtype=torch.bool)
        )
        eh_cond = _to_tt(prompt_embeds, TRANSFORMER_DTYPE)
        eh_uncond = (
            _to_tt(negative_prompt_embeds, TRANSFORMER_DTYPE) if do_cfg else None
        )
        target_size_tt = _to_tt(target_size)
        crop_coords_tt = _to_tt(crop_coords)

        def _dit(hidden, enc, drop, ts):
            return transformer(
                hidden_states=hidden,
                encoder_hidden_states=enc,
                prior_token_id=prior_ids_tt,
                prior_token_drop=drop,
                timestep=ts,
                target_size=target_size_tt,
                crop_coords=crop_coords_tt,
                return_dict=False,
                kv_caches=None,  # t2i: no condition-image KV cache
            )[0]

        # ── Denoising loop (DiT on TT, scheduler on CPU) ───────────────────
        logger.info(f"[STAGE] DiT denoising loop: start ({len(timesteps)} steps)")
        for i, t in enumerate(timesteps):
            logger.info(f"[STEP] DiT step {i + 1}/{len(timesteps)}")
            # Per-step DiT time: the two CFG forwards plus the TT→CPU cast that
            # forces the device sync, so the timer captures real device work.
            t0 = time.perf_counter()
            latent_input = _to_tt(latents, TRANSFORMER_DTYPE)
            timestep = _to_tt(t.expand(B) - 1)

            noise_pred_cond = _to_cpu(
                _dit(latent_input, eh_cond, drop_cond_tt, timestep)
            ).float()
            if do_cfg:
                noise_pred_uncond = _to_cpu(
                    _dit(latent_input, eh_uncond, drop_uncond_tt, timestep)
                ).float()
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                noise_pred = noise_pred_cond
            self._perf["steps"].append(time.perf_counter() - t0)

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        logger.info("[STAGE] DiT denoising loop: done")

        # ── VAE decode (CPU) -> RGB image in [-1, 1] ───────────────────────
        logger.info("[STAGE] VAE decode (CPU): start")
        t0 = time.perf_counter()
        latents = latents.to(vae.dtype)
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.latent_channels, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(vae.config.latents_std)
            .view(1, vae.config.latent_channels, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean
        image = vae.decode(latents, return_dict=False)[0]
        self._perf["components"]["vae"] = time.perf_counter() - t0
        logger.info("[STAGE] VAE decode (CPU): done")

        # Post-process ([-1, 1] -> output_type) via the diffusers image processor.
        image = pipe.image_processor.postprocess(image, output_type=output_type)

        self._perf["total"] = time.perf_counter() - t_total_start
        return image
