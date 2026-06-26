# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""BRIA 2.3 — benchmark-side pipeline for the imagegen harness.

BRIA 2.3 (briaai/BRIA-2.3) is an SDXL-class text-to-image model, so this
mirrors ``playground_v2_5_pipeline.py`` (full SDXL with classifier-free
guidance) almost exactly. Two differences from the other SDXL pipelines:

  - The ``bria_2_3`` loader exposes only the wrapped UNet (its bringup
    contract), not per-component variants. So this pipeline pulls the
    individual ``nn.Module`` components (text_encoder, text_encoder_2, unet,
    vae) out of the loader's ``load_pipe()`` and wraps them locally with the
    same tensor-only contracts the other SDXL pipelines rely on.
  - BRIA sets ``force_zeros_for_empty_prompt = False`` (per the model card),
    so the unconditional branch is produced by *encoding the empty prompt*
    through both text encoders rather than zeroing the embeddings.

Each component can be independently placed on Tenstorrent via
``model.compile(backend="tt") + model.to(xla_device())``. Tokenizers and the
scheduler always stay on CPU. The VAE stays on CPU by default: SDXL VAE on TT
currently hits a GroupNorm/opt-level issue (tt-xla #5176 / #4710), the same
reason ``sdxl_lightning`` keeps its VAE on CPU. Per-component forward+sync
times are collected into ``self._perf`` for the harness to read after each
``generate()`` call.
"""

import time
from typing import Optional

import torch
import torch_xla.core.xla_model as xm
from loguru import logger

from third_party.tt_forge_models.bria_2_3.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)

MODEL_ID = "briaai/BRIA-2.3"
HEIGHT = 1024
WIDTH = 1024


class _TextEncoderWrapper(torch.nn.Module):
    """Return the penultimate hidden state (SDXL cross-attention input)."""

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        out = self.text_encoder(input_ids, output_hidden_states=True)
        return out.hidden_states[-2]


class _TextEncoder2Wrapper(torch.nn.Module):
    """Return (penultimate hidden state, pooled text_embeds)."""

    def __init__(self, text_encoder_2):
        super().__init__()
        self.text_encoder_2 = text_encoder_2

    def forward(self, input_ids):
        out = self.text_encoder_2(input_ids, output_hidden_states=True)
        return out.hidden_states[-2], out.text_embeds


class _UNet2DConditionWrapper(torch.nn.Module):
    """Flatten UNet inputs to positional tensors and return noise_pred."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]


class Bria23Config:
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
        # Harness-set compile options (kept for symmetry with the other SDXL
        # pipelines; the VAE-on-TT opt-level switch is unused while vae_on_tt
        # is False).
        self.compile_options = compile_options or {}


class Bria23Pipeline:
    """BRIA 2.3 (SDXL-class) pipeline with per-component TT toggles."""

    def __init__(self, config: Bria23Config):
        self.config = config

    def setup(self):
        self.load_models()

    def load_models(self):
        # The bria_2_3 loader's load_pipe() returns the full SDXL pipeline on
        # CPU (fp32, eval, requires_grad off, force_zeros_for_empty_prompt
        # already set to False). We pull the components out and wrap them.
        loader = ModelLoader(ModelVariant.BASE)
        pipe = loader._load_pipeline(dtype_override=torch.float32)

        # Tokenizers + scheduler stay on CPU.
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.scheduler = pipe.scheduler

        # For TT-bound components we only register the `tt` dynamo backend here;
        # the actual move to xla_device happens in generate() right before the
        # forward, and we evict back to CPU after. This keeps at most one model
        # resident on TT DRAM at a time.
        self.text_encoder = _TextEncoderWrapper(pipe.text_encoder)
        if self.config.text_encoder_on_tt:
            self.text_encoder.compile(backend="tt")

        self.text_encoder_2 = _TextEncoder2Wrapper(pipe.text_encoder_2)
        if self.config.text_encoder_2_on_tt:
            self.text_encoder_2.compile(backend="tt")

        # UNet runs in bf16 on TT to fit DRAM (fp32 weights are ~10 GB).
        unet = pipe.unet
        if self.config.unet_on_tt:
            unet = unet.to(torch.bfloat16)
        self.unet = _UNet2DConditionWrapper(unet)
        if self.config.unet_on_tt:
            self.unet.compile(backend="tt")

        # VAE stays on CPU by default (see module docstring); keep the raw
        # AutoencoderKL so we can call .decode() directly.
        self.vae = pipe.vae

    def _get_add_time_ids(self, dtype):
        original_size = (self.config.height, self.config.width)
        crops_coords_top_left = (0, 0)
        target_size = (self.config.height, self.config.width)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        return torch.tensor([add_time_ids], dtype=dtype)

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        assert isinstance(
            prompt, str
        ), "Only single-prompt generation (batch_size=1) is tested for now"
        batch_size = 1

        # BRIA sets force_zeros_for_empty_prompt = False, so the unconditional
        # branch is the *encoded empty prompt*, not zeros.
        if negative_prompt is None:
            negative_prompt = ""

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

            # ── Text encoders (encode positive + unconditional) ───────────
            # te1/te2 timers cover both the conditional and unconditional
            # encodes (the real cost of a CFG generation). Each encoder is
            # moved onto the device once, used for both prompts, then evicted.
            logger.info("[STAGE] Text encoders: start")

            # -- Encoder 1 (both prompts) --
            tok1_pos = self.tokenizer(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            tok1_neg = self.tokenizer(
                [negative_prompt],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            if self.config.text_encoder_on_tt:
                self.text_encoder = self.text_encoder.to(device)
                tok1_pos = tok1_pos.to(device)
                tok1_neg = tok1_neg.to(device)
            t0 = time.perf_counter()
            pe1_pos = self.text_encoder(tok1_pos)
            pe1_neg = self.text_encoder(tok1_neg)
            if self.config.text_encoder_on_tt:
                pe1_pos = pe1_pos.to("cpu")
                pe1_neg = pe1_neg.to("cpu")
            self._perf["te1"] = time.perf_counter() - t0
            if self.config.text_encoder_on_tt:
                self.text_encoder = self.text_encoder.to("cpu")

            # -- Encoder 2 (both prompts) --
            tok2_pos = self.tokenizer_2(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            tok2_neg = self.tokenizer_2(
                [negative_prompt],
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            if self.config.text_encoder_2_on_tt:
                self.text_encoder_2 = self.text_encoder_2.to(device)
                tok2_pos = tok2_pos.to(device)
                tok2_neg = tok2_neg.to(device)
            t0 = time.perf_counter()
            pe2_pos, pooled_pos = self.text_encoder_2(tok2_pos)
            pe2_neg, pooled_neg = self.text_encoder_2(tok2_neg)
            if self.config.text_encoder_2_on_tt:
                pe2_pos = pe2_pos.to("cpu")
                pe2_neg = pe2_neg.to("cpu")
                pooled_pos = pooled_pos.to("cpu")
                pooled_neg = pooled_neg.to("cpu")
            self._perf["te2"] = time.perf_counter() - t0
            if self.config.text_encoder_2_on_tt:
                self.text_encoder_2 = self.text_encoder_2.to("cpu")

            logger.info("[STAGE] Text encoders: done")

            # Concat each encoder's hidden states, then CFG-concat (uncond first).
            prompt_embeds_pos = torch.cat([pe1_pos, pe2_pos], dim=-1)
            prompt_embeds_neg = torch.cat([pe1_neg, pe2_neg], dim=-1)
            prompt_embeds = torch.cat([prompt_embeds_neg, prompt_embeds_pos], dim=0)
            add_text_embeds = torch.cat([pooled_neg, pooled_pos], dim=0)

            add_time_ids = self._get_add_time_ids(prompt_embeds.dtype)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

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
            )
            latents = latents * self.scheduler.init_noise_sigma

            # ── Denoising loop (UNet, CFG batch=2) ────────────────────────
            logger.info(
                f"[STAGE] UNet denoising loop: start ({num_inference_steps} steps)"
            )
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

                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                if self.config.unet_on_tt:
                    unet_sample = latent_model_input.to(torch.bfloat16).to(device)
                    unet_t = t.to(torch.bfloat16).to(device)
                else:
                    unet_sample = latent_model_input
                    unet_t = t

                t0 = time.perf_counter()
                noise_pred = self.unet(unet_sample, unet_t, unet_eh, unet_te, unet_ti)
                if self.config.unet_on_tt:
                    noise_pred = noise_pred.to("cpu").to(torch.float32)
                self._perf["unet_steps"].append(time.perf_counter() - t0)

                # CFG combine + scheduler step.
                uncond, text = noise_pred.chunk(2)
                noise_pred = uncond + cfg_scale * (text - uncond)
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
            if self.config.unet_on_tt:
                self.unet = self.unet.to("cpu")
            logger.info("[STAGE] UNet denoising loop: done")

            # ── VAE decode (standard SDXL: divide by scaling_factor) ──────
            # VAE runs on CPU by default; cast latents to the VAE's dtype.
            logger.info("[STAGE] VAE decode: start")
            latents = latents / self.vae.config.scaling_factor
            latents = latents.to(next(self.vae.parameters()).dtype)

            t0 = time.perf_counter()
            image = self.vae.decode(latents, return_dict=False)[0]
            self._perf["vae"] = time.perf_counter() - t0
            logger.info("[STAGE] VAE decode: done")

            self._perf["total"] = time.perf_counter() - t_total_start

            return image
