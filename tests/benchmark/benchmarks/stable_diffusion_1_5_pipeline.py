# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Stable Diffusion 1.5 — benchmark-side pipeline for the imagegen harness.

Mirrors ``sdxl_lightning_pipeline.py`` but for SD1.5's simpler architecture:
a *single* CLIP text encoder (ViT-L/14), one ``UNet2DConditionModel`` and an
``AutoencoderKL`` VAE. The UNet is sourced from the bringup'd
``stable_diffusion_1_5`` loader (which also exposes the matching CLIP text
encoder / tokenizer / scheduler as side effects of ``load_model``); the VAE is
loaded directly from the SD1.5 repo since the loader does not expose a VAE
variant.

Device placement follows the proven SDXL-Lightning config: text encoder + UNet
on Tenstorrent, VAE on CPU (the VAE's GroupNorm decomposition OOMs at
opt_level=0). Each nn.Module is moved CPU→TT→CPU inline at its call site so at
most one heavy net is resident on TT DRAM at a time. Per-component forward+sync
times are collected into ``self._perf`` for the harness to read after each
``generate()`` call.

Note: this bringup baseline runs **without classifier-free guidance** (CFG),
i.e. batch_size=1 and a single conditional UNet forward per step — matching the
existing imagegen pipelines and keeping the compiled shapes conservative.
Production SD1.5 uses CFG (guidance_scale≈7.5), which runs the UNet on a batch
of 2 (uncond+cond) and roughly doubles the per-step UNet cost. See the perf
report for the geometry caveat.
"""

import time
from typing import Optional

import torch
import torch_xla.core.xla_model as xm
from diffusers import AutoencoderKL
from loguru import logger

from third_party.tt_forge_models.stable_diffusion_1_5.pytorch import (
    ModelLoader,
    ModelVariant,
)

# SD1.5 weights repo (UNet, VAE, scheduler live here).
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
HEIGHT = 512
WIDTH = 512


class _TextEncoderWrapper(torch.nn.Module):
    """Returns the CLIP last_hidden_state as a plain tensor (TT-friendly)."""

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        return self.text_encoder(input_ids)[0]


class _UNetWrapper(torch.nn.Module):
    """Returns the predicted noise sample as a plain tensor (TT-friendly)."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(
            sample, timestep, encoder_hidden_states, return_dict=False
        )[0]


class StableDiffusion15Config:
    def __init__(
        self,
        text_encoder_on_tt: bool = True,
        unet_on_tt: bool = True,
        vae_on_tt: bool = False,
        compile_options: Optional[dict] = None,
    ):
        self.model_id = MODEL_ID
        self.width = WIDTH
        self.height = HEIGHT
        self.vae_scale_factor = 8
        self.latents_width = self.width // self.vae_scale_factor
        self.latents_height = self.height // self.vae_scale_factor
        self.text_encoder_on_tt = text_encoder_on_tt
        self.unet_on_tt = unet_on_tt
        self.vae_on_tt = vae_on_tt
        # Harness-set compile options (forwarded for symmetry with SDXL; the
        # SD1.5 VAE stays on CPU by default so no inline opt-level switch).
        self.compile_options = compile_options or {}


class StableDiffusion15Pipeline:
    """Stable Diffusion 1.5 pipeline with per-component TT toggles."""

    def __init__(self, config: StableDiffusion15Config):
        self.config = config

    def setup(self):
        self.load_models()
        self.load_tokenizer_and_scheduler()

    def load_models(self):
        # The SD1.5 loader returns the UNet and, as a side effect, loads the
        # matching CLIP text encoder / tokenizer / scheduler onto the loader
        # instance. We reuse those so the perf path tracks the bringup'd loader.
        loader = ModelLoader(ModelVariant.BASE)

        # UNet runs in bf16 on TT to match TT execution / fit DRAM.
        unet_dtype = torch.bfloat16 if self.config.unet_on_tt else torch.float32
        unet = loader.load_model(dtype_override=unet_dtype)
        self.in_channels = loader.in_channels

        self._loader = loader
        self.tokenizer = loader.tokenizer
        self.scheduler = loader.scheduler

        # Text encoder stays fp32 (CLIP); wrap to return a plain tensor.
        self.text_encoder = _TextEncoderWrapper(loader.text_encoder)
        if self.config.text_encoder_on_tt:
            self.text_encoder.compile(backend="tt")

        self.unet = _UNetWrapper(unet)
        if self.config.unet_on_tt:
            self.unet.compile(backend="tt")

        # VAE is not exposed by the loader; load it directly. fp32 on CPU.
        self.vae = AutoencoderKL.from_pretrained(
            self.config.model_id, subfolder="vae", torch_dtype=torch.float32
        )
        if self.config.vae_on_tt:
            self.vae.compile(backend="tt")

    def load_tokenizer_and_scheduler(self):
        # Tokenizer + scheduler already loaded by the loader; nothing extra to do.
        # (kept as a named step to mirror the SDXL pipeline structure)
        pass

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        assert isinstance(
            prompt, str
        ), "Only single-prompt generation (batch_size=1) is tested for now"
        batch_size = 1

        device = xm.xla_device()
        # Per-component forward+sync times (reset every generate() call). SD1.5
        # has a single text encoder, so te2 is always 0.0 (kept for the harness
        # which sums te1+te2+unet+vae).
        self._perf = {
            "te1": None,
            "te2": 0.0,
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

            # ── Text encoder (CLIPTextModel, no CFG) ──────────────────────
            logger.info("[STAGE] Text encoder: start")
            tokens = self.tokenizer(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device="cpu")

            # CPU → TT
            if self.config.text_encoder_on_tt:
                self.text_encoder = self.text_encoder.to(device)
                tokens = tokens.to(device=device)

            t0 = time.perf_counter()
            prompt_embeds = self.text_encoder(tokens)
            # TT → CPU (cpu cast forces sync — timer ends after this)
            if self.config.text_encoder_on_tt:
                prompt_embeds = prompt_embeds.to("cpu")
            self._perf["te1"] = time.perf_counter() - t0

            if self.config.text_encoder_on_tt:
                self.text_encoder = self.text_encoder.to("cpu")

            logger.info("[STAGE] Text encoder: done")

            # ── Timesteps ─────────────────────────────────────────────────
            self.scheduler.set_timesteps(num_inference_steps, device="cpu")
            timesteps = self.scheduler.timesteps

            # ── Latents ───────────────────────────────────────────────────
            latent_shape = (
                batch_size,
                self.in_channels,
                self.config.latents_height,
                self.config.latents_width,
            )
            latents = torch.randn(
                latent_shape, generator=generator, dtype=torch.float32
            ).to(device="cpu")
            latents = latents * self.scheduler.init_noise_sigma

            # ── Denoising loop (UNet, no CFG) ─────────────────────────────
            logger.info(
                f"[STAGE] UNet denoising loop: start ({num_inference_steps} steps)"
            )
            if self.config.unet_on_tt:
                self.unet = self.unet.to(device)

            # Loop-invariant input: convert encoder_hidden_states once.
            if self.config.unet_on_tt:
                unet_eh = prompt_embeds.to(torch.bfloat16).to(device)
            else:
                unet_eh = prompt_embeds

            for i, t in enumerate(timesteps):
                logger.info(f"[STEP] UNet step {i + 1}/{num_inference_steps}")

                latent_model_input = self.scheduler.scale_model_input(latents, t)

                # CPU → TT (UNet runs in bf16 on TT). Only sample + timestep
                # change per step; embeds are hoisted above.
                if self.config.unet_on_tt:
                    unet_sample = latent_model_input.to(torch.bfloat16).to(device)
                    unet_t = t.to(torch.bfloat16).to(device)
                else:
                    unet_sample = latent_model_input
                    unet_t = t

                t0 = time.perf_counter()
                noise_pred = self.unet(unet_sample, unet_t, unet_eh)
                # TT → CPU (cpu cast forces sync — timer ends after this)
                if self.config.unet_on_tt:
                    noise_pred = noise_pred.to("cpu").to(torch.float32)
                self._perf["unet_steps"].append(time.perf_counter() - t0)

                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
            if self.config.unet_on_tt:
                self.unet = self.unet.to("cpu")
            logger.info("[STAGE] UNet denoising loop: done")

            # ── VAE decode (standard SD: divide by scaling_factor) ────────
            logger.info("[STAGE] VAE decode: start")
            latents = latents / self.vae.config.scaling_factor

            if self.config.vae_on_tt:
                self.vae = self.vae.to(device)
                latents = latents.to(device)

            t0 = time.perf_counter()
            image = self.vae.decode(latents, return_dict=False)[0]
            # TT → CPU (cpu cast forces sync — timer ends after this)
            if self.config.vae_on_tt:
                image = image.to("cpu")
            self._perf["vae"] = time.perf_counter() - t0

            if self.config.vae_on_tt:
                self.vae = self.vae.to("cpu")

            logger.info("[STAGE] VAE decode: done")
            self._perf["total"] = time.perf_counter() - t_total_start

            return image
