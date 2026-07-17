# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Z-Image — single-chip benchmark-side pipeline for the imagegen harness.

Mirrors the nightly e2e test (tests/torch/models/z_image/test_z_image_pipeline.py) but
instruments per-component timings into ``self._perf`` for the harness. Every
compute module runs on one Blackhole chip, compiled with
``torch.compile(backend="tt")``:

  - text encoder (Qwen3)                 → components["text_encoder"]
  - transformer (ZImageTransformer2DModel) denoise loop → steps
  - VAE decoder (AutoencoderKL)          → components["vae"]
"""

import gc
import time
from typing import Optional

import torch
import torch_xla
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
    TEXT_TOKEN_MAX_LEN,
    VAE_SCALE_FACTOR,
    WIDTH,
    load_text_encoder,
    load_transformer,
    load_vae,
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
        vae_tiling: bool = True,
    ):
        self.height = height
        self.width = width
        self.vae_scale_factor = VAE_SCALE_FACTOR
        # Forwarded for parity with the other imagegen pipelines; unused inline.
        self.compile_options = compile_options or {}
        # Tiled VAE decode keeps the 1280x720 decode activations small so the
        # host-side spike during decode stays bounded. Flip off to revert to a
        # single full-frame decode.
        self.vae_tiling = vae_tiling


class ZImagePipeline_TT:
    """Z-Image text-to-image pipeline with every module on a single TT chip."""

    def __init__(self, config: ZImageConfig):
        self.config = config
        self._perf = {}

    def setup(self):
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            REPO_ID, subfolder="scheduler"
        )
        self._device = torch_xla.device()
        self._tokenizer = self._load_tokenizer()

        # NOTE (host RAM): the raw modules are intentionally NOT preloaded here.
        # generate() loads each component from disk right before its stage, moves
        # it to the device, compiles it, runs it, then fully frees it (del + gc)
        # before the next stage. This bounds *host* RAM to ~max(component) instead
        # of sum(components): the Qwen3 encoder (~7.5 GB), ~6.2B transformer
        # (~12 GB) and VAE never sit in host memory at once, and in particular the
        # encoder + transformer are gone before the VAE decode. Reload cost is a
        # few seconds from the page cache and lands outside the measured device
        # windows, so it does not affect reported perf. On a single Blackhole the
        # same eviction already keeps at most one component on TT DRAM at a time.

    def _load_tokenizer(self):
        """Load the Qwen tokenizer from the local HF snapshot.

        model_utils.load_tokenizer() resolves it by repo id, which forces an
        online HEAD for a tokenizer/config.json that does not exist in that
        subfolder — fine online (404 → fallback) but a hard failure whenever the
        box's flaky network is down. Resolving the cached snapshot path first and
        loading from there is offline-safe and never touches the network.
        """
        from transformers import AutoTokenizer

        try:
            from huggingface_hub import snapshot_download

            snap = snapshot_download(
                REPO_ID, allow_patterns=["tokenizer/*"], local_files_only=True
            )
            return AutoTokenizer.from_pretrained(snap, subfolder="tokenizer")
        except Exception:
            # Fall back to the repo-id path (needs network) if the cache lookup
            # misses for any reason.
            return AutoTokenizer.from_pretrained(REPO_ID, subfolder="tokenizer")

    def _tokenize(self, prompt: str):
        """Chat-template + tokenize a prompt (mirrors model_utils.tokenize_prompt)."""
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        text_inputs = self._tokenizer(
            [text],
            padding="max_length",
            max_length=TEXT_TOKEN_MAX_LEN,
            truncation=True,
            return_tensors="pt",
        )
        return text_inputs.input_ids, text_inputs.attention_mask

    def _encode(self, prompt: str, encoder) -> torch.Tensor:
        input_ids, attention_mask = self._tokenize(prompt)
        hidden = encoder(
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
            # ── Text encoder (Qwen3) → prompt embeds, then free ──────────
            # Loaded here (not in setup) and fully released at the end of the
            # stage so its ~7.5 GB never overlaps the transformer or VAE on host.
            logger.info("[STAGE] Text encoder: start")
            text_encoder = _TextEncoderWrapper(load_text_encoder(DTYPE)).eval()
            text_encoder = text_encoder.to(self._device)
            te_compiled = torch.compile(text_encoder, backend="tt")
            t0 = time.perf_counter()
            cap_pos = self._encode(prompt, te_compiled)
            cap_neg = self._encode(NEGATIVE_PROMPT, te_compiled) if do_cfg else None
            self._perf["components"]["text_encoder"] = time.perf_counter() - t0
            del te_compiled, text_encoder
            gc.collect()
            torch_xla.sync()
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

            # ── Denoising loop (transformer) ─────
            logger.info(
                f"[STAGE] Transformer denoising loop: start "
                f"({num_inference_steps} steps)"
            )
            transformer = _TransformerWrapper(load_transformer(DTYPE)).eval()
            transformer = transformer.to(self._device)
            tf_compiled = torch.compile(transformer, backend="tt")
            for i, t in enumerate(timesteps):
                logger.info(f"[STEP] Transformer step {i + 1}/{num_inference_steps}")
                timestep = ((1000 - t.expand(1)) / 1000).to(DTYPE)
                latent_input = latents.to(DTYPE)

                t0 = time.perf_counter()
                pos = self._forward(tf_compiled, latent_input, timestep, cap_pos)
                if do_cfg:
                    neg = self._forward(tf_compiled, latent_input, timestep, cap_neg)
                    pred = pos + GUIDANCE_SCALE * (pos - neg)
                else:
                    pred = pos
                self._perf["steps"].append(time.perf_counter() - t0)

                noise_pred = (-pred).squeeze(2)
                latents = self.scheduler.step(
                    noise_pred.to(torch.float32), t, latents, return_dict=False
                )[0]
            # Free the transformer (~12 GB) + its compiled graph before the VAE
            # decode so the decode stage runs with only the VAE resident on host.
            del tf_compiled, transformer
            gc.collect()
            torch_xla.sync()
            logger.info("[STAGE] Transformer denoising loop: done")

            # ── VAE decode → raw pixels in [-1, 1], then free ────────────
            logger.info("[STAGE] VAE decode: start")
            vae_wrapper = _VaeDecodeWrapper(load_vae(DTYPE)).eval()
            if self.config.vae_tiling and hasattr(vae_wrapper.vae, "enable_tiling"):
                # Tiled decode bounds the 1280x720 decode activations (and their
                # host staging) to a single tile instead of the full frame.
                vae_wrapper.vae.enable_tiling()
            vae_wrapper = vae_wrapper.to(self._device)
            vae_compiled = torch.compile(vae_wrapper, backend="tt")
            t0 = time.perf_counter()
            image = vae_compiled(latents.to(self._device)).cpu().float()
            self._perf["components"]["vae"] = time.perf_counter() - t0
            del vae_compiled, vae_wrapper
            gc.collect()
            torch_xla.sync()
            logger.info("[STAGE] VAE decode: done")

        self._perf["total"] = time.perf_counter() - t_total_start
        return image

    def _forward(self, transformer, latents, timestep, cap_feats) -> torch.Tensor:
        out = transformer(
            latents.to(self._device),
            timestep.to(self._device),
            cap_feats.to(self._device),
        )
        return out.cpu().float()  # forces sync
