# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Lumina-Image-2.0 — benchmark-side pipeline for the imagegen harness.

Mirrors ``sdxl_lightning_pipeline.py``: each nn.Module component
(text_encoder, transformer, vae) is independently movable to Tenstorrent via
``model.compile(backend="tt") + model.to(xla_device())``. Tokenizer and
scheduler always stay on CPU. CPU->TT->CPU device switching is done inline at
the call site of each nn component so at most one model is resident on TT DRAM
at a time. Per-component forward+sync times are collected into ``self._perf``
for the harness to read after each ``generate()`` call.

The generation faithfully follows ``diffusers.Lumina2Pipeline``:
  - text encoder is Gemma2; the conditioning is ``hidden_states[-2]`` (not the
    final hidden state),
  - classifier-free guidance runs the transformer twice per step
    (cond + uncond) with normalization-based guidance,
  - flow-match (FlowMatchEuler) scheduler with shift ``mu`` derived from the
    image sequence length,
  - Lumina uses t=0 as noise and t=1 as image, so the timestep is reversed and
    the predicted velocity is negated before the scheduler step.

Lumina has a single text encoder, so the harness's ``te2`` slot is reported as
0.0. The VAE decode at 1024x1024 needs a single ~8.6 GB DRAM intermediate that
OOMs a single Wormhole chip, so the VAE runs on CPU by default (same choice as
SDXL-Lightning); ``vae_on_tt`` is left as a toggle for later perf tuning.
"""

import time
from typing import Optional

import numpy as np
import torch
import torch_xla.core.xla_model as xm
from diffusers import FlowMatchEulerDiscreteScheduler
from loguru import logger
from transformers import AutoTokenizer

from third_party.tt_forge_models.lumina_image.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.lumina_image.pytorch.src.model_utils import (
    LUMINA_REPO_ID,
)

MODEL_ID = LUMINA_REPO_ID
HEIGHT = 1024
WIDTH = 1024
MAX_SEQUENCE_LENGTH = 256
GUIDANCE_SCALE = 4.0

# System prompt prepended to the positive prompt (verbatim from Lumina2Pipeline).
SYSTEM_PROMPT = (
    "You are an assistant designed to generate superior images with the superior "
    "degree of image-text alignment based on textual prompts or user prompts."
)


def _calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """Flow-match timestep shift ``mu`` (copied from Lumina2Pipeline)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class _GemmaHiddenStates(torch.nn.Module):
    """Run Gemma2 and return ``hidden_states[-2]`` as a plain tensor.

    Lumina2Pipeline conditions on the second-to-last Gemma2 hidden state, not
    the final one. ``use_cache=False`` is pinned for the same reason as the
    loader's wrapper: Gemma-2's sliding-window cache slice exceeds the tt-mlir
    slice bound (tenstorrent/tt-xla#4900); a single encode pass needs no cache.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return out.hidden_states[-2]


class LuminaImage2Config:
    def __init__(
        self,
        text_encoder_on_tt: bool = True,
        transformer_on_tt: bool = True,
        vae_on_tt: bool = False,
        guidance_scale: float = GUIDANCE_SCALE,
        compile_options: Optional[dict] = None,
    ):
        self.model_id = MODEL_ID
        self.width = WIDTH
        self.height = HEIGHT
        self.vae_scale_factor = 8
        self.max_sequence_length = MAX_SEQUENCE_LENGTH
        self.guidance_scale = guidance_scale
        self.text_encoder_on_tt = text_encoder_on_tt
        self.transformer_on_tt = transformer_on_tt
        self.vae_on_tt = vae_on_tt
        # Harness-set compile options; preserved around any inline opt switch.
        self.compile_options = compile_options or {}


class LuminaImage2Pipeline:
    """Lumina-Image-2.0 pipeline with per-component TT toggles."""

    def __init__(self, config: LuminaImage2Config):
        self.config = config

    def setup(self):
        self.load_models()
        self.load_scheduler()
        self.load_tokenizer()

    def load_models(self):
        # Load all models on CPU. For TT-bound components we only register the
        # `tt` dynamo backend here; the actual move to xla_device happens in
        # generate() right before the forward, evicted back to CPU after, so at
        # most one model is resident on TT DRAM at a time.

        # Text encoder (Gemma2) in bf16 on TT. Wrap the loaded encoder so the
        # forward returns hidden_states[-2] (Lumina's conditioning tensor).
        te_wrapper = ModelLoader(ModelVariant.TEXT_ENCODER).load_model(
            dtype_override=torch.bfloat16
        )
        self.text_encoder = _GemmaHiddenStates(te_wrapper.encoder)
        if self.config.text_encoder_on_tt:
            self.text_encoder.compile(backend="tt")

        # Transformer (Lumina2 Next-DiT) in bf16 on TT.
        self.transformer = ModelLoader(ModelVariant.TRANSFORMER).load_model(
            dtype_override=torch.bfloat16
        )
        if self.config.transformer_on_tt:
            self.transformer.compile(backend="tt")

        # VAE decoder on CPU in fp32 (1024^2 decode OOMs a single chip).
        self.vae = ModelLoader(ModelVariant.VAE).load_model(
            dtype_override=torch.float32
        )
        if self.config.vae_on_tt:
            self.vae.compile(backend="tt")

    def load_scheduler(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.model_id, subfolder="scheduler"
        )
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, subfolder="tokenizer"
        )
        self.tokenizer.padding_side = "right"

    def _encode(self, text, device):
        """Tokenize ``text`` and return (embeds, attention_mask) on CPU.

        Runs the Gemma2 encoder on TT (if enabled) and brings the result back
        to CPU. ``embeds`` is hidden_states[-2] cast to bf16.
        """
        text_inputs = self.tokenizer(
            [text],
            padding="max_length",
            max_length=self.config.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        if self.config.text_encoder_on_tt:
            ids_dev = input_ids.to(device)
            mask_dev = attention_mask.to(device)
            embeds = self.text_encoder(ids_dev, mask_dev).to("cpu")
        else:
            embeds = self.text_encoder(input_ids, attention_mask)

        return embeds.to(torch.bfloat16), attention_mask

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        assert isinstance(
            prompt, str
        ), "Only single-prompt generation (batch_size=1) is tested for now"
        batch_size = 1
        cfg = self.config
        guidance_scale = cfg.guidance_scale
        do_cfg = guidance_scale > 1

        device = xm.xla_device()
        # Per-component forward+sync times (reset every generate() call).
        self._perf = {
            "te1": 0.0,
            "te2": 0.0,  # Lumina has a single text encoder.
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

            # ── Text encoder (Gemma2) ─────────────────────────────────────
            logger.info("[STAGE] Text encoder: start")
            if cfg.text_encoder_on_tt:
                self.text_encoder = self.text_encoder.to(device)

            t0 = time.perf_counter()
            # Positive prompt gets the system prompt prepended; the negative
            # ("") is encoded raw — matching Lumina2Pipeline.encode_prompt.
            pos_text = SYSTEM_PROMPT + " <Prompt Start> " + prompt
            prompt_embeds, prompt_attention_mask = self._encode(pos_text, device)
            if do_cfg:
                neg_embeds, neg_attention_mask = self._encode("", device)
            self._perf["te1"] = time.perf_counter() - t0

            if cfg.text_encoder_on_tt:
                self.text_encoder = self.text_encoder.to("cpu")
            logger.info("[STAGE] Text encoder: done")

            # ── Latents ───────────────────────────────────────────────────
            # VAE applies 8x compression and packing requires latent H/W to be
            # divisible by 2 → 2 * (dim // 16).
            latent_h = 2 * (cfg.height // (cfg.vae_scale_factor * 2))
            latent_w = 2 * (cfg.width // (cfg.vae_scale_factor * 2))
            latent_channels = self.transformer.transformer.config.in_channels
            latents = torch.randn(
                (batch_size, latent_channels, latent_h, latent_w),
                generator=generator,
                dtype=torch.float32,
            )

            # ── Timesteps (flow-match with image-seq-len shift) ───────────
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            image_seq_len = latents.shape[1]
            mu = _calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            self.scheduler.set_timesteps(sigmas=sigmas, device="cpu", mu=mu)
            timesteps = self.scheduler.timesteps

            # ── Denoising loop (transformer, CFG) ─────────────────────────
            logger.info(
                f"[STAGE] Transformer denoising loop: start ({num_inference_steps} steps)"
            )
            if cfg.transformer_on_tt:
                self.transformer = self.transformer.to(device)

            # Loop-invariant conditioning: move to device once, reuse per step.
            if cfg.transformer_on_tt:
                eh_cond = prompt_embeds.to(device)
                mask_cond = prompt_attention_mask.to(device)
                if do_cfg:
                    eh_uncond = neg_embeds.to(device)
                    mask_uncond = neg_attention_mask.to(device)
            else:
                eh_cond, mask_cond = prompt_embeds, prompt_attention_mask
                if do_cfg:
                    eh_uncond, mask_uncond = neg_embeds, neg_attention_mask

            for i, t in enumerate(timesteps):
                logger.info(f"[STEP] Transformer step {i + 1}/{num_inference_steps}")

                # Lumina uses t=0 as noise and t=1 as image, so reverse t.
                current_timestep = 1 - t / self.num_train_timesteps
                current_timestep = current_timestep.expand(latents.shape[0])

                if cfg.transformer_on_tt:
                    sample = latents.to(torch.bfloat16).to(device)
                    ts = current_timestep.to(torch.float32).to(device)
                else:
                    sample = latents.to(torch.bfloat16)
                    ts = current_timestep.to(torch.float32)

                t0 = time.perf_counter()
                noise_pred_cond = self.transformer(sample, ts, eh_cond, mask_cond)
                if cfg.transformer_on_tt:
                    noise_pred_cond = noise_pred_cond.to("cpu")
                noise_pred_cond = noise_pred_cond.to(torch.float32)

                if do_cfg:
                    noise_pred_uncond = self.transformer(
                        sample, ts, eh_uncond, mask_uncond
                    )
                    if cfg.transformer_on_tt:
                        noise_pred_uncond = noise_pred_uncond.to("cpu")
                    noise_pred_uncond = noise_pred_uncond.to(torch.float32)
                self._perf["unet_steps"].append(time.perf_counter() - t0)

                if do_cfg:
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                    # Normalization-based guidance scale.
                    cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
                    noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_pred = noise_pred * (cond_norm / noise_norm)
                else:
                    noise_pred = noise_pred_cond

                # Velocity is negated before the scheduler step (Lumina).
                noise_pred = -noise_pred
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

            if cfg.transformer_on_tt:
                self.transformer = self.transformer.to("cpu")
            logger.info("[STAGE] Transformer denoising loop: done")

            # ── VAE decode ────────────────────────────────────────────────
            logger.info("[STAGE] VAE decode: start")
            latents = (
                latents / self.vae.vae.config.scaling_factor
            ) + self.vae.vae.config.shift_factor

            if cfg.vae_on_tt:
                self.vae = self.vae.to(device)
                vae_in = latents.to(torch.bfloat16).to(device)
            else:
                vae_in = latents.to(torch.float32)

            t0 = time.perf_counter()
            image = self.vae(vae_in)
            if cfg.vae_on_tt:
                image = image.to("cpu")
            self._perf["vae"] = time.perf_counter() - t0

            if cfg.vae_on_tt:
                self.vae = self.vae.to("cpu")
            logger.info("[STAGE] VAE decode: done")

            self._perf["total"] = time.perf_counter() - t_total_start
            return image
