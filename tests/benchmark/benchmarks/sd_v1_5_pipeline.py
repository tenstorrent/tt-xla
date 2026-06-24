# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Stable Diffusion 1.5 — benchmark-side pipeline for the imagegen harness.

Mirrors ``sdxl_lightning_pipeline.py`` but for the classic SD1.5 architecture:
a single CLIP text encoder (``CLIPTextModel``), a ``UNet2DConditionModel`` and
an ``AutoencoderKL`` VAE, driven by classifier-free guidance (CFG).

The UNet, text encoder, tokenizer and scheduler come from the tt-forge-models
SD1.5 loader (``stable_diffusion_1_5``). That loader exposes only the UNet (plus
the tokenizer/text-encoder/scheduler it sets as attributes), so the VAE — which
the loader does not return — is loaded here from the same HF repo, exactly as the
SDXL pipelines load their extra components on the benchmark side.

Each nn.Module component (text_encoder, unet, vae) can be independently moved to
Tenstorrent via ``model.compile(backend="tt") + model.to(xla_device())``.
Tokenizer and scheduler always stay on CPU. CPU→TT→CPU device switching is done
inline at the call site of each component. Per-component forward+sync times are
collected into ``self._perf`` for the harness to read after each generate() call.
"""

import time
from typing import Optional

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from diffusers import AutoencoderKL
from loguru import logger

from third_party.tt_forge_models.stable_diffusion_1_5.pytorch import (
    ModelLoader,
    ModelVariant,
)

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
HEIGHT = 512
WIDTH = 512
# SD1.5 base is trained with classifier-free guidance; 7.5 is the canonical scale.
GUIDANCE_SCALE = 7.5


# ---------------------------------------------------------------------------
# Pure-tensor wrappers (flatten HF dict/object outputs so the tt backend sees
# tensor-in / tensor-out forwards).
# ---------------------------------------------------------------------------


class TextEncoderWrapper(torch.nn.Module):
    """Return the last hidden state of the CLIP text encoder.

    SD1.5 cross-attention consumes the full last hidden state (``output[0]``),
    unlike SDXL which uses the penultimate layer.
    """

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        return self.text_encoder(input_ids)[0]


class UNet2DConditionWrapper(torch.nn.Module):
    """Flatten UNet2DConditionModel inputs/outputs to pure tensors."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


class VAEDecoderWrapper(torch.nn.Module):
    """Expose only the decoder half of AutoencoderKL."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


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
        self.guidance_scale = GUIDANCE_SCALE
        self.text_encoder_on_tt = text_encoder_on_tt
        self.unet_on_tt = unet_on_tt
        self.vae_on_tt = vae_on_tt
        # Harness-set compile options; used to preserve them around the
        # VAE-only opt_level switch (only relevant if vae_on_tt is True).
        self.compile_options = compile_options or {}


class StableDiffusion15Pipeline:
    """SD1.5 pipeline with per-component TT toggles and classifier-free guidance."""

    def __init__(self, config: StableDiffusion15Config):
        self.config = config

    def setup(self):
        self.load_models()
        self.load_scheduler_and_tokenizer()

    def load_models(self):
        # The loader returns the UNet and sets ``.text_encoder`` / ``.tokenizer``
        # / ``.scheduler`` as attributes. UNet runs in bf16 on TT to fit DRAM;
        # the CLIP text encoder stays fp32 (matches the SDXL pipelines).
        self.loader = ModelLoader(ModelVariant.BASE)
        unet_dtype = torch.bfloat16 if self.config.unet_on_tt else torch.float32
        raw_unet = self.loader.load_model(dtype_override=unet_dtype)

        self.text_encoder = TextEncoderWrapper(self.loader.text_encoder)
        if self.config.text_encoder_on_tt:
            self.text_encoder.compile(backend="tt")

        self.unet = UNet2DConditionWrapper(raw_unet)
        if self.config.unet_on_tt:
            self.unet.compile(backend="tt")

        # VAE is not exposed by the loader — load it from the same HF repo.
        vae = AutoencoderKL.from_pretrained(
            self.config.model_id, subfolder="vae", torch_dtype=torch.float32
        ).eval()
        self.vae = VAEDecoderWrapper(vae)
        if self.config.vae_on_tt:
            self.vae.compile(backend="tt")

    def load_scheduler_and_tokenizer(self):
        # Scheduler (LMSDiscreteScheduler) and tokenizer come from the loader.
        self.scheduler = self.loader.scheduler
        self.tokenizer = self.loader.tokenizer

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
        # has a single text encoder, so te2 is always 0 for this model.
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

            # ── Text encoder (CLIPTextModel) ──────────────────────────────
            # Tokenize the unconditional ("") and conditional prompts together
            # so a single batch-2 forward yields both CFG branches.
            logger.info("[STAGE] Text encoder: start")
            tokens = self.tokenizer(
                ["", prompt],
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
            text_embeddings = self.text_encoder(tokens)  # (2, 77, 768)
            # TT → CPU (cpu cast forces sync — timer ends after this)
            if self.config.text_encoder_on_tt:
                text_embeddings = text_embeddings.to("cpu")
            self._perf["te1"] = time.perf_counter() - t0

            if self.config.text_encoder_on_tt:
                self.text_encoder = self.text_encoder.to("cpu")
            text_embeddings = text_embeddings.to(torch.float32)
            logger.info("[STAGE] Text encoder: done")

            # ── Timesteps ─────────────────────────────────────────────────
            self.scheduler.set_timesteps(num_inference_steps)
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

            # ── Denoising loop (UNet, batched CFG) ────────────────────────
            logger.info(
                f"[STAGE] UNet denoising loop: start ({num_inference_steps} steps)"
            )
            # Move UNet to TT once before the loop; evict after.
            if self.config.unet_on_tt:
                self.unet = self.unet.to(device)
                unet_eh = text_embeddings.to(torch.bfloat16).to(device)
            else:
                unet_eh = text_embeddings

            for i, t in enumerate(timesteps):
                logger.info(f"[STEP] UNet step {i + 1}/{num_inference_steps}")

                # CFG: duplicate latents into [uncond, cond] (batch 2).
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

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

                # CFG combine.
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
            # Evict UNet from TT before VAE comes onto the device.
            if self.config.unet_on_tt:
                self.unet = self.unet.to("cpu")
            logger.info("[STAGE] UNet denoising loop: done")

            # ── VAE decode (SD1.5: divide by scaling_factor) ──────────────
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
