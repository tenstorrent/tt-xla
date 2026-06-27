# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SRPO (FLUX.1-dev fine-tune) — benchmark-side pipeline for the imagegen harness.

SRPO ships only the FLUX transformer weights; the SRPO loader rebuilds the FLUX
transformer and assembles a ``FluxPipeline`` around the FLUX.1-dev CLIP/T5 text
encoders, tokenizers, scheduler and a tiny (``taef1``) VAE. This pipeline drives
that loader for the imagegen benchmark harness.

Component placement (bringup-safe defaults):
  - transformer (the ~11.9B FLUX denoiser) runs on Tenstorrent — it is the heavy
    per-step net and the only TT-resident component.
  - CLIP + T5 text encoders and the tiny VAE run on CPU. They are cheap relative
    to the denoising loop and keep a single large model resident on TT DRAM.

The denoising loop mirrors ``FluxPipeline.__call__`` (flow-matching schedule with
``calculate_shift`` / ``retrieve_timesteps``); the transformer is invoked directly
with device-resident inputs, exactly as ``sdxl_lightning_pipeline.py`` drives the
UNet. Per-component forward+sync times are collected into ``self._perf`` for the
harness to read after each ``generate()`` call.
"""

import time
from typing import Optional

import numpy as np
import torch
import torch_xla.core.xla_model as xm
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from loguru import logger

from third_party.tt_forge_models.srpo.pytorch.loader import ModelLoader, ModelVariant

HEIGHT = 1024
WIDTH = 1024
MAX_SEQUENCE_LENGTH = 512


class SRPOConfig:
    def __init__(
        self,
        transformer_on_tt: bool = True,
        height: int = HEIGHT,
        width: int = WIDTH,
        guidance_scale: float = 3.5,
        compile_options: Optional[dict] = None,
    ):
        self.transformer_on_tt = transformer_on_tt
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
        # Harness-set compile options (kept for parity with the other pipelines;
        # SRPO does not switch options inline since only the transformer is on TT).
        self.compile_options = compile_options or {}


class SRPOPipeline:
    """SRPO (FLUX) pipeline: transformer on TT, text encoders + VAE on CPU."""

    def __init__(self, config: SRPOConfig):
        self.config = config

    def setup(self):
        # Build the SRPO transformer and the surrounding FluxPipeline (encoders /
        # tokenizers / scheduler / tiny VAE). Everything is bf16 to fit TT DRAM
        # and keep CPU<->TT casts trivial.
        loader = ModelLoader(ModelVariant.DEV)
        self.transformer = loader.load_model(dtype_override=torch.bfloat16)
        self.pipe = loader.pipe  # full FluxPipeline, all components on CPU

        if self.config.transformer_on_tt:
            self.transformer.compile(backend="tt")

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 20,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        assert isinstance(
            prompt, str
        ), "Only single-prompt generation (batch_size=1) is tested for now"
        batch_size = 1
        num_images_per_prompt = 1
        device = xm.xla_device()

        # Per-component forward+sync times (reset every generate() call).
        self._perf = {
            "te1": None,
            "te2": None,
            "unet_steps": [],
            "vae": None,
            "total": None,
        }
        t_total_start = time.perf_counter()

        with torch.no_grad():
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)
            else:
                generator.seed()

            # ── Text encoder 1 (CLIP, pooled) on CPU ──────────────────────
            logger.info("[STAGE] Text encoder 1 (CLIP): start")
            t0 = time.perf_counter()
            pooled_prompt_embeds = self.pipe._get_clip_prompt_embeds(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                device="cpu",
            )
            self._perf["te1"] = time.perf_counter() - t0
            logger.info("[STAGE] Text encoder 1 (CLIP): done")

            # ── Text encoder 2 (T5) on CPU ────────────────────────────────
            logger.info("[STAGE] Text encoder 2 (T5): start")
            t0 = time.perf_counter()
            prompt_embeds = self.pipe._get_t5_prompt_embeds(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
                device="cpu",
            )
            self._perf["te2"] = time.perf_counter() - t0
            logger.info("[STAGE] Text encoder 2 (T5): done")

            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
                dtype=prompt_embeds.dtype
            )

            # ── Latents (CPU) ─────────────────────────────────────────────
            num_channels_latents = self.transformer.config.in_channels // 4
            latents, latent_image_ids = self.pipe.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                self.config.height,
                self.config.width,
                prompt_embeds.dtype,
                "cpu",
                generator,
                None,
            )

            # ── Timesteps (flow-matching schedule, CPU) ───────────────────
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            image_seq_len = latents.shape[1]
            mu = calculate_shift(
                image_seq_len,
                self.pipe.scheduler.config.get("base_image_seq_len", 256),
                self.pipe.scheduler.config.get("max_image_seq_len", 4096),
                self.pipe.scheduler.config.get("base_shift", 0.5),
                self.pipe.scheduler.config.get("max_shift", 1.15),
            )
            timesteps, num_inference_steps = retrieve_timesteps(
                self.pipe.scheduler,
                num_inference_steps,
                "cpu",
                sigmas=sigmas,
                mu=mu,
            )
            self.pipe.scheduler.set_begin_index(0)

            # ── Guidance (FLUX.1-dev uses distilled-guidance embeds) ───────
            if self.transformer.config.guidance_embeds:
                guidance = torch.full(
                    [1], self.config.guidance_scale, dtype=torch.float32
                ).expand(latents.shape[0])
            else:
                guidance = None

            # ── Denoising loop (transformer on TT) ────────────────────────
            logger.info(
                f"[STAGE] FLUX denoising loop: start ({num_inference_steps} steps)"
            )
            if self.config.transformer_on_tt:
                self.transformer = self.transformer.to(device)
                # Loop-invariant inputs: move to TT once and reuse every step.
                tt_pooled = pooled_prompt_embeds.to(torch.bfloat16).to(device)
                tt_eh = prompt_embeds.to(torch.bfloat16).to(device)
                tt_txt_ids = text_ids.to(torch.bfloat16).to(device)
                tt_img_ids = latent_image_ids.to(torch.bfloat16).to(device)
                tt_guidance = (
                    guidance.to(torch.bfloat16).to(device)
                    if guidance is not None
                    else None
                )
            else:
                tt_pooled = pooled_prompt_embeds
                tt_eh = prompt_embeds
                tt_txt_ids = text_ids
                tt_img_ids = latent_image_ids
                tt_guidance = guidance

            for i, t in enumerate(timesteps):
                logger.info(f"[STEP] FLUX step {i + 1}/{num_inference_steps}")
                timestep = (t.expand(latents.shape[0]).to(latents.dtype)) / 1000

                # CPU → TT (only latents + timestep change per step).
                if self.config.transformer_on_tt:
                    tt_hidden = latents.to(torch.bfloat16).to(device)
                    tt_timestep = timestep.to(torch.bfloat16).to(device)
                else:
                    tt_hidden = latents
                    tt_timestep = timestep

                t0 = time.perf_counter()
                noise_pred = self.transformer(
                    hidden_states=tt_hidden,
                    timestep=tt_timestep,
                    guidance=tt_guidance,
                    pooled_projections=tt_pooled,
                    encoder_hidden_states=tt_eh,
                    txt_ids=tt_txt_ids,
                    img_ids=tt_img_ids,
                    joint_attention_kwargs={},
                    return_dict=False,
                )[0]
                # TT → CPU (cpu cast forces sync — timer ends after this).
                if self.config.transformer_on_tt:
                    noise_pred = noise_pred.to("cpu").to(torch.float32)
                self._perf["unet_steps"].append(time.perf_counter() - t0)

                latents = self.pipe.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

            if self.config.transformer_on_tt:
                self.transformer = self.transformer.to("cpu")
            logger.info("[STAGE] FLUX denoising loop: done")

            # ── VAE decode (tiny taef1 VAE, on CPU) ───────────────────────
            logger.info("[STAGE] VAE decode: start")
            latents = self.pipe._unpack_latents(
                latents,
                self.config.height,
                self.config.width,
                self.pipe.vae_scale_factor,
            )
            latents = (
                latents / self.pipe.vae.config.scaling_factor
            ) + self.pipe.vae.config.shift_factor
            # scheduler.step promotes latents to float32 (noise_pred is cast to
            # float32 for CPU stepping); match the bf16 VAE before decode.
            latents = latents.to(self.pipe.vae.dtype)

            t0 = time.perf_counter()
            image = self.pipe.vae.decode(latents, return_dict=False)[0]
            self._perf["vae"] = time.perf_counter() - t0
            logger.info("[STAGE] VAE decode: done")

            self._perf["total"] = time.perf_counter() - t_total_start

            return image
