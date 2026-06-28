# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SDXL-Lightning — benchmark-side pipeline for the imagegen harness.

Each nn.Module component (text_encoder, text_encoder_2, unet, vae) can be
independently moved to Tenstorrent via
`model.compile(backend="tt") + model.to(xla_device())`. Tokenizer and
scheduler always stay on CPU. CPU→TT→CPU device switching is done inline at
the call site of each nn component. Per-component forward+sync times are
collected into `self._perf` for the harness to read after each generate() call.
"""

import time
from typing import Optional

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from diffusers import EulerDiscreteScheduler
from loguru import logger
from transformers import CLIPTokenizer

from third_party.tt_forge_models.sdxl_lightning.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.sdxl_lightning.pytorch.src.model_utils import (
    SDXL_BASE_REPO_ID,
)

MODEL_ID = SDXL_BASE_REPO_ID
HEIGHT = 1024
WIDTH = 1024


class SDXLLightningConfig:
    def __init__(
        self,
        text_encoder_on_tt: bool = True,
        text_encoder_2_on_tt: bool = True,
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
        self.text_encoder_2_on_tt = text_encoder_2_on_tt
        self.unet_on_tt = unet_on_tt
        self.vae_on_tt = vae_on_tt
        # Harness-set compile options; used to preserve them around the
        # VAE-only opt_level switch (only relevant if vae_on_tt is True).
        self.compile_options = compile_options or {}


class SDXLLightningPipeline:
    """SDXL-Lightning pipeline with per-component TT toggles."""

    def __init__(self, config: SDXLLightningConfig):
        self.config = config

    def setup(self):
        self.load_models()
        self.load_scheduler()
        self.load_tokenizers()

    def load_models(self):
        # Load all models on CPU. For TT-bound components we only register the
        # `tt` dynamo backend here; the actual move to xla_device happens in
        # generate() right before the forward, and we evict back to CPU after.
        # This keeps at most one model resident on TT DRAM at a time.
        self.text_encoder = ModelLoader(ModelVariant.TEXT_ENCODER).load_model(
            dtype_override=torch.float32
        )
        if self.config.text_encoder_on_tt:
            self.text_encoder.compile(backend="tt")

        self.text_encoder_2 = ModelLoader(ModelVariant.TEXT_ENCODER_2).load_model(
            dtype_override=torch.float32
        )
        if self.config.text_encoder_2_on_tt:
            self.text_encoder_2.compile(backend="tt")

        # UNet runs in bf16 on TT to fit DRAM (fp32 weights ~10.3 GB).
        unet_dtype = torch.bfloat16 if self.config.unet_on_tt else torch.float32
        self.unet = ModelLoader(ModelVariant.UNET).load_model(dtype_override=unet_dtype)
        if self.config.unet_on_tt:
            self.unet.compile(backend="tt")

        self.vae = ModelLoader(ModelVariant.VAE).load_model(
            dtype_override=torch.float32
        )
        if self.config.vae_on_tt:
            self.vae.compile(backend="tt")

    def load_scheduler(self):
        # SDXL-Lightning requires Euler with "trailing" timestep spacing.
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            self.config.model_id, subfolder="scheduler", timestep_spacing="trailing"
        )

    def load_tokenizers(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.model_id, subfolder="tokenizer"
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.config.model_id, subfolder="tokenizer_2"
        )

    def _get_add_time_ids(self, dtype):
        original_size = (self.config.height, self.config.width)
        crops_coords_top_left = (0, 0)
        target_size = (self.config.height, self.config.width)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        return torch.tensor([add_time_ids], dtype=dtype)

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 4,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        assert isinstance(
            prompt, str
        ), "Only single-prompt generation (batch_size=1) is tested for now"
        batch_size = 1

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

            # ── Text encoder 1 (CLIPTextModel) ────────────────────────────
            logger.info("[STAGE] Text encoder 1: start")
            tokens_1 = self.tokenizer(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device="cpu")

            # CPU → TT
            if self.config.text_encoder_on_tt:
                self.text_encoder = self.text_encoder.to(device)
                tokens_1 = tokens_1.to(device=device)

            t0 = time.perf_counter()
            prompt_embeds_1 = self.text_encoder(tokens_1)
            # TT → CPU (cpu cast forces sync — timer ends after this)
            if self.config.text_encoder_on_tt:
                prompt_embeds_1 = prompt_embeds_1.to("cpu")
            self._perf["te1"] = time.perf_counter() - t0

            if self.config.text_encoder_on_tt:
                self.text_encoder = self.text_encoder.to("cpu")

            logger.info("[STAGE] Text encoder 1: done")

            # ── Text encoder 2 (CLIPTextModelWithProjection) ──────────────
            logger.info("[STAGE] Text encoder 2: start")
            tokens_2 = self.tokenizer_2(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device="cpu")

            # CPU → TT
            if self.config.text_encoder_2_on_tt:
                self.text_encoder_2 = self.text_encoder_2.to(device)
                tokens_2 = tokens_2.to(device=device)

            t0 = time.perf_counter()
            prompt_embeds_2, pooled_prompt_embeds = self.text_encoder_2(tokens_2)
            # TT → CPU (cpu cast forces sync — timer ends after this)
            if self.config.text_encoder_2_on_tt:
                prompt_embeds_2 = prompt_embeds_2.to("cpu")
                pooled_prompt_embeds = pooled_prompt_embeds.to("cpu")
            self._perf["te2"] = time.perf_counter() - t0

            if self.config.text_encoder_2_on_tt:
                self.text_encoder_2 = self.text_encoder_2.to("cpu")

            logger.info("[STAGE] Text encoder 2: done")

            # Concat the two encoders' hidden states (no CFG: batch stays 1).
            prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
            add_text_embeds = pooled_prompt_embeds  # (1, 1280)

            add_time_ids = self._get_add_time_ids(prompt_embeds.dtype).to(
                "cpu"
            )  # (1, 6)

            # ── Timesteps ─────────────────────────────────────────────────
            self.scheduler.set_timesteps(num_inference_steps, device="cpu")
            timesteps = self.scheduler.timesteps

            # ── Latents ───────────────────────────────────────────────────
            latent_shape = (
                batch_size,
                4,
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
            # Move UNet to TT once before the loop; evict after.
            if self.config.unet_on_tt:
                self.unet = self.unet.to(device)

            # Loop-invariant inputs: convert once, reuse across all steps.
            if self.config.unet_on_tt:
                unet_eh = prompt_embeds.to(torch.bfloat16).to(device)
                unet_te = add_text_embeds.to(torch.bfloat16).to(device)
                unet_ti = add_time_ids.to(torch.bfloat16).to(device)
            else:
                unet_eh = prompt_embeds
                unet_te = add_text_embeds
                unet_ti = add_time_ids

            for i, t in enumerate(timesteps):
                logger.info(f"[STEP] UNet step {i + 1}/{num_inference_steps}")

                latent_model_input = self.scheduler.scale_model_input(latents, t)

                # CPU → TT (UNet runs in bf16 on TT). Only sample + timestep
                # change per step; embeds/time_ids are hoisted above.
                if self.config.unet_on_tt:
                    unet_sample = latent_model_input.to(torch.bfloat16).to(device)
                    unet_t = t.to(torch.bfloat16).to(device)
                else:
                    unet_sample = latent_model_input
                    unet_t = t

                t0 = time.perf_counter()
                noise_pred = self.unet(unet_sample, unet_t, unet_eh, unet_te, unet_ti)
                # TT → CPU (cpu cast forces sync — timer ends after this)
                if self.config.unet_on_tt:
                    noise_pred = noise_pred.to("cpu").to(torch.float32)
                self._perf["unet_steps"].append(time.perf_counter() - t0)

                # No CFG combine (guidance_scale=0): use noise_pred directly.
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
            # Evict UNet from TT before VAE comes onto the device.
            if self.config.unet_on_tt:
                self.unet = self.unet.to("cpu")
            logger.info("[STAGE] UNet denoising loop: done")

            # ── VAE decode (standard SDXL: divide by scaling_factor) ──────
            logger.info("[STAGE] VAE decode: start")
            latents = latents / self.vae.vae.config.scaling_factor

            # opt_level=1 (composite ttnn.group_norm) is only needed when the VAE
            # runs on TT; by default it runs on CPU.
            if self.config.vae_on_tt:
                torch_xla.set_custom_compile_options(
                    {**self.config.compile_options, "optimization_level": 1}
                )
                self.vae = self.vae.to(device)
                latents = latents.to(device)

            t0 = time.perf_counter()
            image = self.vae(latents)
            # TT → CPU (cpu cast forces sync — timer ends after this)
            if self.config.vae_on_tt:
                image = image.to("cpu")
            self._perf["vae"] = time.perf_counter() - t0

            if self.config.vae_on_tt:
                self.vae = self.vae.to("cpu")
                # Restore harness-set options (un-merge the VAE opt_level bump).
                torch_xla.set_custom_compile_options(self.config.compile_options)

            logger.info("[STAGE] VAE decode: done")
            self._perf["total"] = time.perf_counter() - t_total_start

            return image
