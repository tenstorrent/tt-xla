# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Z-Image — single-chip benchmark-side pipeline for the imagegen harness.

Mirrors the nightly e2e test (tests/torch/models/z_image/test_pipeline.py) but
instruments per-component timings into ``self._perf`` for the harness. Every
compute module runs on one Blackhole chip, compiled with
``torch.compile(backend="tt")``:

  - text encoder (Qwen3)                 → components["text_encoder"]
  - transformer (ZImageTransformer2DModel) denoise loop → steps
  - VAE decoder (AutoencoderKL)          → components["vae"]

Classifier-free guidance runs cond + uncond as two separate batch=1 passes
(the transformer's complex RoPE only legalizes at batch=1 on the pinned
tt-mlir; batch=2 fails, tt-mlir #8874). Z-Image has a single text encoder, so
it reports only the components it runs — the harness drops absent components
from the report (model-agnostic ``components``/``steps`` schema).
"""

import gc
import time
from typing import Optional

import torch
import torch_xla.core.xla_model as xm
from diffusers import FlowMatchEulerDiscreteScheduler
from loguru import logger

from third_party.tt_forge_models.z_image.pytorch.src.model_utils import (
    DTYPE,
    GUIDANCE_SCALE,
    HEIGHT,
    LATENT_CHANNELS,
    NEGATIVE_PROMPT,
    REPO_ID,
    SEED,
    VAE_SCALE_FACTOR,
    WIDTH,
    load_text_encoder,
    load_transformer,
    load_vae,
    tokenize_prompt,
)


def _calculate_shift(image_seq_len, base_seq=256, max_seq=4096, base=0.5, max_=1.15):
    m = (max_ - base) / (max_seq - base_seq)
    return image_seq_len * m + (base - m * base_seq)


class _TextEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return out.hidden_states[-2]


class _TransformerWrapper(torch.nn.Module):
    """One batch=1 transformer pass; cap_feats is (L, D)."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, latents, timestep, cap_feats):
        x_list = list(latents.unsqueeze(2).unbind(dim=0))
        t = timestep.reshape(-1).to(dtype=latents.dtype)
        out = self.transformer(x_list, t, [cap_feats], return_dict=False)[0]
        return torch.stack([o.float() for o in out], dim=0)


class _VaeDecodeWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        z = latents.to(dtype=self.vae.dtype)
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]


class ZImageConfig:
    def __init__(
        self,
        height: int = HEIGHT,
        width: int = WIDTH,
        compile_options: Optional[dict] = None,
    ):
        self.height = height
        self.width = width
        self.vae_scale_factor = VAE_SCALE_FACTOR
        # Forwarded for parity with the other imagegen pipelines; unused inline.
        self.compile_options = compile_options or {}


class ZImagePipeline_TT:
    """Z-Image text-to-image pipeline with every module on a single TT chip."""

    def __init__(self, config: ZImageConfig):
        self.config = config
        self._perf = {}

    def setup(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            REPO_ID, subfolder="scheduler"
        )
        self._device = xm.xla_device()

        # Load on CPU and only register the tt backend here. Each component is
        # moved to device in generate() right before it runs and evicted after,
        # so peak DRAM ~= max(component) not sum — the Qwen3 encoder + ~6.2B
        # transformer + VAE do not all sit on the single chip at once.
        self.text_encoder = _TextEncoderWrapper(load_text_encoder(DTYPE)).eval()
        self.transformer = _TransformerWrapper(load_transformer(DTYPE)).eval()
        self.vae = _VaeDecodeWrapper(load_vae(DTYPE)).eval()

        self.text_encoder.compile(backend="tt")
        self.transformer.compile(backend="tt")
        self.vae.compile(backend="tt")

    def _encode(self, prompt: str) -> torch.Tensor:
        input_ids, attention_mask = tokenize_prompt(prompt)
        hidden = self.text_encoder(
            input_ids.to(self._device), attention_mask.bool().to(self._device)
        )
        hidden = hidden.cpu()  # forces sync
        mask = attention_mask[0].bool()
        return hidden[0][mask].to(DTYPE)

    def generate(
        self,
        prompt: str,
        num_inference_steps: int,
        seed: Optional[int] = SEED,
    ) -> torch.Tensor:
        do_cfg = GUIDANCE_SCALE > 0
        self._perf = {
            "components": {},
            "steps": [],
            "step_metric_name": "transformer_step",
            "total": None,
        }
        t_total_start = time.perf_counter()

        with torch.no_grad():
            # ── Text encoder (Qwen3) → prompt embeds, then evict ─────────
            logger.info("[STAGE] Text encoder: start")
            self.text_encoder = self.text_encoder.to(self._device)
            t0 = time.perf_counter()
            cap_pos = self._encode(prompt)
            cap_neg = self._encode(NEGATIVE_PROMPT) if do_cfg else None
            self._perf["components"]["text_encoder"] = time.perf_counter() - t0
            self.text_encoder = self.text_encoder.to("cpu")
            gc.collect()
            logger.info("[STAGE] Text encoder: done")

            # ── Latents (fp32 on CPU) ────────────────────────────────────
            vsf = self.config.vae_scale_factor
            latent_h = 2 * (int(self.config.height) // (vsf * 2))
            latent_w = 2 * (int(self.config.width) // (vsf * 2))
            generator = torch.Generator(device="cpu").manual_seed(seed or SEED)
            latents = torch.randn(
                (1, LATENT_CHANNELS, latent_h, latent_w),
                generator=generator,
                dtype=torch.float32,
            )

            # ── Timesteps (resolution-dependent mu shift) ────────────────
            image_seq_len = (latent_h // 2) * (latent_w // 2)
            mu = _calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            self.scheduler.sigma_min = 0.0
            set_ts_kwargs = {}
            import inspect

            if "mu" in inspect.signature(self.scheduler.set_timesteps).parameters:
                set_ts_kwargs["mu"] = mu
            self.scheduler.set_timesteps(
                num_inference_steps, device="cpu", **set_ts_kwargs
            )
            self.scheduler.set_begin_index(0)
            timesteps = self.scheduler.timesteps

            # ── Denoising loop (transformer; CFG = 2 batch=1 passes) ─────
            logger.info(
                f"[STAGE] Transformer denoising loop: start "
                f"({num_inference_steps} steps)"
            )
            self.transformer = self.transformer.to(self._device)
            for i, t in enumerate(timesteps):
                logger.info(f"[STEP] Transformer step {i + 1}/{num_inference_steps}")
                timestep = ((1000 - t.expand(1)) / 1000).to(DTYPE)
                latent_input = latents.to(DTYPE)

                t0 = time.perf_counter()
                pos = self._forward(latent_input, timestep, cap_pos)
                if do_cfg:
                    neg = self._forward(latent_input, timestep, cap_neg)
                    pred = pos + GUIDANCE_SCALE * (pos - neg)
                else:
                    pred = pos
                self._perf["steps"].append(time.perf_counter() - t0)

                noise_pred = (-pred).squeeze(2)
                latents = self.scheduler.step(
                    noise_pred.to(torch.float32), t, latents, return_dict=False
                )[0]
            self.transformer = self.transformer.to("cpu")
            gc.collect()
            logger.info("[STAGE] Transformer denoising loop: done")

            # ── VAE decode → raw pixels in [-1, 1], then evict ───────────
            logger.info("[STAGE] VAE decode: start")
            self.vae = self.vae.to(self._device)
            t0 = time.perf_counter()
            image = self.vae(latents.to(self._device)).cpu().float()
            self._perf["components"]["vae"] = time.perf_counter() - t0
            self.vae = self.vae.to("cpu")
            gc.collect()
            logger.info("[STAGE] VAE decode: done")

        self._perf["total"] = time.perf_counter() - t_total_start
        return image

    def _forward(self, latents, timestep, cap_feats) -> torch.Tensor:
        out = self.transformer(
            latents.to(self._device),
            timestep.to(self._device),
            cap_feats.to(self._device),
        )
        return out.cpu().float()  # forces sync
