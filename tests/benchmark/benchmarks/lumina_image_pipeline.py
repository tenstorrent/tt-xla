# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Lumina-Image-2.0 — benchmark-side pipeline for the imagegen harness.

Lumina-Image-2.0 is a flow-matching text-to-image diffusion model whose three
learned components all run **tensor-parallel** across a multi-chip mesh (the
``("batch", "model")`` mesh built in ``shard_to_tt``): the Gemma-2 text encoder,
the ``Lumina2Transformer2DModel`` DiT and the AutoencoderKL decoder. Tokenizer,
FlowMatchEuler scheduler, latent sampling and the CFG combine stay on CPU. The
end-to-end pipeline (device split + per-component sharding) lives in
``tt_forge_models`` and is the same one the nightly pipeline test drives.

The imagegen harness (``benchmarks/imagegen_benchmark.py``) needs two things the
base ``LuminaImagePipeline`` does not provide:

  - a ``generate(prompt, num_inference_steps, ...)`` that accepts the step count
    per call (warmup uses 1 step, steady-state uses the full count -- the base
    reads it from ``config.num_inference_steps``, fixed at ``setup`` time), and
  - a ``self._perf`` dict populated with per-component / per-step timings.

So -- like the GLM-Image wrapper next door -- this module subclasses the
``tt_forge_models`` pipeline and overrides ``generate`` to add both, without
duplicating the model loading / sharding setup. The denoising body mirrors the
base pipeline's ``generate`` exactly; the only changes are the
``num_inference_steps`` override, the timing hooks, the raw-decode return value
(``[-1, 1]``, what ``utils.save_image`` expects) and the VAE compile-option
scope described in ``generate``.
"""

import time
from typing import Optional

import numpy as np
import torch
import torch_xla.core.xla_model as xm
from loguru import logger

from third_party.tt_forge_models.lumina_image.pytorch.src.pipeline import (
    HEIGHT,
    NEGATIVE_PROMPT,
    PROMPT,
    SEED,
    WIDTH,
    LuminaImageConfig,
)
from third_party.tt_forge_models.lumina_image.pytorch.src.pipeline import (
    LuminaImagePipeline as _BaseLuminaImagePipeline,
)

__all__ = [
    "LuminaImageConfig",
    "LuminaImagePipeline",
    "PROMPT",
    "NEGATIVE_PROMPT",
    "SEED",
    "HEIGHT",
    "WIDTH",
]


class LuminaImagePipeline(_BaseLuminaImagePipeline):
    """Lumina-Image-2.0 pipeline with benchmark-harness timing instrumentation.

    Reuses the base pipeline's model loading, per-component tensor-parallel
    sharding and ``tt``-backend compilation (``setup`` / ``load_models`` /
    ``shard_to_tt``) and only overrides ``setup`` (to expose ``mesh_shape``) and
    ``generate`` (to accept a per-call ``num_inference_steps`` and to populate
    ``self._perf``).
    """

    def setup(self):
        """Run the base setup, then expose the mesh shape to the harness.

        ``imagegen_benchmark`` reads ``pipeline.mesh_shape`` for the perf report;
        the base pipeline only keeps the ``Mesh`` object (``self.mesh``), so
        without this the report records a null mesh shape for a run that is in
        fact tensor-parallel. Stays ``None`` on an unsharded / CPU-only run,
        where the base pipeline builds no mesh at all.
        """
        super().setup()
        mesh = getattr(self, "mesh", None)
        self.mesh_shape = tuple(mesh.mesh_shape) if mesh is not None else None
        logger.info(f"[setup] mesh_shape={self.mesh_shape}")

    @torch.no_grad()
    def generate(
        self,
        prompt: str = PROMPT,
        negative_prompt: str = NEGATIVE_PROMPT,
        seed: Optional[int] = SEED,
        num_inference_steps: Optional[int] = None,
    ):
        """Run the Lumina-Image-2.0 t2i pipeline and record per-stage timings.

        Mirrors ``tt_forge_models`` ``LuminaImagePipeline.generate`` (upstream
        ``Lumina2Pipeline.__call__`` with a CPU/TT split: Gemma-2 encode, DiT
        denoise and VAE decode on TT; tokenizer, scheduler, latent sampling and
        CFG combine on CPU), adding:

          - ``num_inference_steps``: overrides ``config.num_inference_steps`` for
            this call (the harness warms up with 1 step, then runs the full
            count).
          - ``self._perf``: harness-readable timings. ``components`` holds scalar
            per-stage seconds (``text_encode`` -- both CFG branches -- and
            ``vae``), ``steps`` holds per-DiT-step seconds (the two CFG forwards
            plus the ``_cpu`` cast after them, which forces the device sync so
            the timer captures real device time), ``step_metric_name`` labels the
            per-step metric, and ``total`` is the full ``generate`` wall time.

        One further deviation from the base: the VAE decode is scoped in
        ``_vae_compile_options()``. The base pipeline defines that helper (and
        ``VAE_OPT_LEVEL``) but never enters it, so its decode compiles at
        whatever optimization level the caller installed. That is harmless in the
        nightly test, which leaves the level at the tt-mlir default, but the
        benchmark drives the level explicitly, and the 1024x1024 decode needs
        ``ttir.group_norm`` kept intact (tt-xla #4710). Entering the scope here
        makes the decode's compile options independent of the harness setting,
        and the ``_cpu`` sync happens inside the block so the graph is lowered
        under it.

        Returns the raw VAE decode as a ``(1, 3, H, W)`` tensor in ``[-1, 1]`` --
        the range ``utils.save_image`` expects. The base's ``output_type``
        post-processing is skipped rather than re-exposed: the harness has one
        consumer for this value, so a second meaning for ``"latent"`` / ``"pt"``
        would only be a trap.
        """
        from diffusers.pipelines.lumina2.pipeline_lumina2 import (
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

        cfg = self.config
        pipe = self.pipe
        scheduler = self.scheduler
        do_cfg = cfg.do_classifier_free_guidance
        height, width = cfg.height, cfg.width
        num_inference_steps = num_inference_steps or cfg.num_inference_steps
        # Host device for everything outside the three TT stages.
        cpu = torch.device("cpu")

        # 1. Check inputs. Raise error if not correct.
        pipe.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            max_sequence_length=cfg.max_sequence_length,
        )

        # 2. Define call parameters.
        batch_size = 1

        # ── Gemma-2 text encode (TT) ───────────────────────────────────────
        # Upstream ``encode_prompt``; the Gemma-2 forward lands on TT via the
        # base pipeline's ``_get_gemma_prompt_embeds`` override. The system
        # prompt is prefixed to the positive prompt only (upstream behavior);
        # the negative prompt is encoded raw. Timed as one component: with CFG on
        # this is two encoder forwards, each already synced back to CPU by
        # ``_encode_on_tt``.
        logger.info("[STAGE] Gemma-2 text encode (TT): start")
        t0 = time.perf_counter()
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = pipe.encode_prompt(
            prompt,
            do_cfg,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            device=cpu,
            max_sequence_length=cfg.max_sequence_length,
        )
        self._perf["components"]["text_encode"] = time.perf_counter() - t0
        logger.info("[STAGE] Gemma-2 text encode (TT): done")

        if (
            do_cfg
            and cfg.pad_negative_caption
            and negative_prompt_attention_mask is not None
        ):
            negative_prompt_attention_mask = self._pad_caption_mask(
                negative_prompt_attention_mask, prompt_attention_mask
            )

        # 3. Prepare latents. Sampled on CPU as upstream does (randn_tensor with
        #    a CPU generator), so the values are seed-identical to a CPU run.
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        latents = pipe.prepare_latents(
            batch_size,
            pipe.transformer.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            cpu,
            generator,
        )

        # 4. Prepare timesteps (flow-match scheduler, CPU).
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        # NOTE: upstream passes latents.shape[1] (the latent channel count) here,
        # not the patched image sequence length. Kept verbatim so the shift --
        # and hence the whole timestep schedule -- matches the reference.
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        # Upstream also pins the timestep schedule to CPU when XLA is available.
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, cpu, sigmas=sigmas, mu=mu
        )
        num_train_timesteps = scheduler.config.num_train_timesteps

        # Loop-invariant conditioning: cast to TT once, reused every step.
        prompt_embeds_tt = self._to_stage("transformer", prompt_embeds)
        prompt_mask_tt = self._to_stage("transformer", prompt_attention_mask)
        if do_cfg:
            negative_prompt_embeds_tt = self._to_stage(
                "transformer", negative_prompt_embeds
            )
            negative_prompt_mask_tt = self._to_stage(
                "transformer", negative_prompt_attention_mask
            )

        # 5. Denoising loop (DiT on TT, scheduler on CPU).
        logger.info(f"[STAGE] DiT denoising loop: start ({num_inference_steps} steps)")
        for i, t in enumerate(timesteps):
            logger.info(
                f"[STEP] denoise {i + 1}/{num_inference_steps} (t={float(t):.4f})"
            )
            # compute whether apply classifier-free truncation on this timestep
            do_classifier_free_truncation = (
                i + 1
            ) / num_inference_steps > cfg.cfg_trunc_ratio
            # reverse the timestep since Lumina uses t=0 as the noise and t=1 as
            # the image
            current_timestep = 1 - t / num_train_timesteps
            # broadcast to batch dimension
            current_timestep = current_timestep.expand(latents.shape[0])

            # Per-step DiT time: the CFG forwards plus the TT→CPU casts that
            # force the device sync, so the timer captures real device work.
            t0 = time.perf_counter()
            timestep_tt = self._to_stage("transformer", current_timestep)
            latents_tt = self._to_stage("transformer", latents)

            noise_pred_cond = self._cpu(
                self.transformer(
                    latents_tt, timestep_tt, prompt_embeds_tt, prompt_mask_tt
                )
            ).float()

            # perform normalization-based guidance scale on a truncated timestep
            # interval
            if do_cfg and not do_classifier_free_truncation:
                noise_pred_uncond = self._cpu(
                    self.transformer(
                        latents_tt,
                        timestep_tt,
                        negative_prompt_embeds_tt,
                        negative_prompt_mask_tt,
                    )
                ).float()
                self._perf["steps"].append(time.perf_counter() - t0)
                noise_pred = noise_pred_uncond + cfg.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                # apply normalization after classifier-free guidance
                if cfg.cfg_normalization:
                    cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
                    noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_pred = noise_pred * (cond_norm / noise_norm)
            else:
                self._perf["steps"].append(time.perf_counter() - t0)
                noise_pred = noise_pred_cond

            # compute the previous noisy sample x_t -> x_t-1
            noise_pred = -noise_pred
            latents = scheduler.step(
                noise_pred.to(latents.dtype), t, latents, return_dict=False
            )[0]

            if cfg.on_tt:
                xm.mark_step()
        logger.info("[STAGE] DiT denoising loop: done")

        # 6. VAE decode (TT) -> RGB image in [-1, 1].
        logger.info("[STAGE] VAE decode (TT): start")
        t0 = time.perf_counter()
        latents = (
            latents / pipe.vae.config.scaling_factor
        ) + pipe.vae.config.shift_factor
        with self._vae_compile_options():
            image = self._cpu(self.vae(self._to_stage("vae", latents)))
        self._perf["components"]["vae"] = time.perf_counter() - t0
        logger.info("[STAGE] VAE decode (TT): done")

        self._perf["total"] = time.perf_counter() - t_total_start
        return image
