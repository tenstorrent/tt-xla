# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev — multichip benchmark-side pipeline for the imagegen harness.

Mirrors the nightly e2e test (tests/torch/models/flux2/test_flux2_pipeline.py):
the diffusers Flux2Pipeline orchestrates, the Mistral3 text encoder and Flux2
transformer are SPMD-sharded (model axis), the VAE is replicated. The text
encoder is evicted before the transformer is placed and the VAE is placed
lazily, so peak DRAM ≈ max(component). Per-component times go into ``self._perf``.
"""

import os
import time
from typing import Optional

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import Flux2Pipeline
from loguru import logger
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.flux2.pytorch.src.model_utils import (
    DTYPE,
    GUIDANCE_SCALE,
    HEIGHT,
    MESH_NAMES,
    MESH_SHAPES,
    PROMPT,
    REPO_ID,
    SEED,
    WIDTH,
    Mistral3TextEncoderWrapper,
    shard_text_encoder_specs,
    shard_transformer_specs,
    tokenize_prompt,
)


class _DeviceDenoiser:
    """Routes the transformer to the TP-sharded model on TT; each call is one
    denoise step, timed into ``perf["unet_steps"]``."""

    def __init__(self, transformer, mesh, perf):
        self._dev = torch_xla.device()
        self._perf = perf
        self.config = transformer.config
        self.dtype = next(transformer.parameters()).dtype

        transformer = transformer.to(self._dev)
        if hasattr(transformer, "tie_weights"):
            transformer.tie_weights()
        for tensor, spec in shard_transformer_specs(transformer).items():
            xs.mark_sharding(tensor, mesh, spec)
        self._compiled = torch.compile(transformer, backend="tt")

    def __call__(self, **kwargs):
        moved = {
            k: (v.to(self._dev) if torch.is_tensor(v) else v) for k, v in kwargs.items()
        }
        t0 = time.perf_counter()
        out = self._compiled(**moved)
        torch_xla.sync()
        # cpu cast forces a sync — the timer ends after the result lands on host.
        if isinstance(out, (tuple, list)):
            result = type(out)(o.cpu() if torch.is_tensor(o) else o for o in out)
        else:
            result = out.cpu()
        self._perf["unet_steps"].append(time.perf_counter() - t0)
        return result


class _DeviceVAEDecoder:
    """Routes vae.decode() to TT (replicated), placed lazily. Decode time goes
    into ``perf["vae"]``; raw pixels ([-1, 1]) are stashed on ``last_pixels`` so
    the harness can save them without the pipeline's PIL postprocess."""

    def __init__(self, vae, mesh, perf):
        self._dev = torch_xla.device()
        self._perf = perf
        self.config = vae.config
        self.dtype = next(vae.parameters()).dtype
        self.bn = vae.bn  # stays on CPU; pipeline reads it host-side for denorm
        self._vae = vae
        self._compiled = None
        self.last_pixels = None

    def decode(self, latents, return_dict=False):
        # Lazy device placement: keep the VAE off-device during the denoise loop
        # so it does not inflate the denoiser's peak DRAM; place it only now.
        if self._compiled is None:
            vae = self._vae.to(self._dev)
            self._compiled = torch.compile(
                lambda z: vae.decode(z, return_dict=False)[0], backend="tt"
            )
        t0 = time.perf_counter()
        out = self._compiled(latents.to(self._dev))
        torch_xla.sync()
        image = out.cpu() if torch.is_tensor(out) else out
        self._perf["vae"] = time.perf_counter() - t0
        self.last_pixels = image
        return (image,)


class Flux2Config:
    def __init__(
        self,
        height: int = HEIGHT,
        width: int = WIDTH,
        compile_options: Optional[dict] = None,
    ):
        self.height = height
        self.width = width
        # Forwarded for parity with the other imagegen pipelines; unused inline.
        self.compile_options = compile_options or {}


class Flux2Pipeline_TT:
    """Flux2Pipeline with every module on TT, tensor-parallel sharded. Built
    once; generate() runs twice (warmup + steady). Raw transformer/VAE are kept
    so the TT wrappers can be rebuilt fresh each call."""

    def __init__(self, config: Flux2Config):
        self.config = config
        self._perf = {}

    def setup(self):
        # CONVERT_SHLO_TO_SHARDY=1 + use_spmd() (mirrors infra.enable_spmd):
        # required so tt-mlir gets shardy annotations, else presharded args lose
        # their @Sharding custom call and compilation fails.
        os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
        xr.use_spmd()
        self.num_devices = xr.global_runtime_device_count()
        self.mesh_shape = MESH_SHAPES[self.num_devices]
        device_ids = np.array(range(self.num_devices))
        self.mesh = Mesh(device_ids, self.mesh_shape, MESH_NAMES)
        logger.info(
            f"FLUX.2 mesh {self.mesh_shape} (names={MESH_NAMES}) "
            f"over {self.num_devices} devices"
        )
        self.pipe = Flux2Pipeline.from_pretrained(REPO_ID, torch_dtype=DTYPE)
        # Keep the raw modules so wrappers can be rebuilt on each generate().
        self._raw_transformer = self.pipe.transformer
        self._raw_vae = self.pipe.vae

    def generate(
        self,
        prompt: str,
        num_inference_steps: int,
        seed: Optional[int] = SEED,
    ):
        dev = torch_xla.device()
        self._perf = {
            "te1": None,
            "unet_steps": [],
            "vae": None,
            "total": None,
            "step_label": "Transformer",
        }
        t_total_start = time.perf_counter()

        # ── Stage 1: text encoder (sharded, compiled) → prompt embeds, evict ──
        logger.info("[STAGE] Text encoder: start")
        text_encoder = self.pipe.text_encoder
        encoder_wrapper = Mistral3TextEncoderWrapper(text_encoder).eval()
        input_ids, attention_mask = tokenize_prompt(prompt)

        text_encoder = text_encoder.to(dev)
        if hasattr(text_encoder, "tie_weights"):
            text_encoder.tie_weights()
        te_specs = shard_text_encoder_specs(text_encoder)
        assert te_specs, "text-encoder shard spec is empty — descent failed (would OOM)"
        for tensor, spec in te_specs.items():
            xs.mark_sharding(tensor, self.mesh, spec)
        te_compiled = torch.compile(encoder_wrapper, backend="tt")

        t0 = time.perf_counter()
        with torch.no_grad():
            prompt_embeds = te_compiled(input_ids.to(dev), attention_mask.to(dev))
        torch_xla.sync()
        prompt_embeds = prompt_embeds.cpu()
        self._perf["te1"] = time.perf_counter() - t0

        # Free the 24B encoder from device before placing the 32B denoiser.
        self.pipe.text_encoder = text_encoder.to("cpu")
        del te_compiled, encoder_wrapper
        import gc

        gc.collect()
        torch_xla.sync()
        logger.info("[STAGE] Text encoder: done")

        # ── Stage 2: denoiser (sharded) + VAE (replicated, lazy) → image ─────
        logger.info("[STAGE] Transformer + VAE: start")
        self.pipe.transformer = _DeviceDenoiser(
            self._raw_transformer, self.mesh, self._perf
        )
        vae_wrapper = _DeviceVAEDecoder(self._raw_vae, self.mesh, self._perf)
        self.pipe.vae = vae_wrapper

        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        self.pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            height=self.config.height,
            width=self.config.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        )
        logger.info("[STAGE] Transformer + VAE: done")

        self._perf["total"] = time.perf_counter() - t_total_start
        # Raw VAE pixels in [-1, 1], shape (1, 3, H, W) — the harness's
        # save_image() expects this range, so return it instead of the PIL.
        return vae_wrapper.last_pixels
