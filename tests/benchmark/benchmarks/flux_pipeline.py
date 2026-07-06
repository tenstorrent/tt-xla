# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1-dev — multichip benchmark-side pipeline for the imagegen harness.

Mirrors the nightly e2e test (tests/torch/models/flux/test_flux_pipeline.py):
the diffusers ``FluxPipeline`` orchestrates, the CLIP + T5 text encoders and the
VAE are replicated, and the FluxTransformer2DModel is SPMD-sharded (model axis)
across a 4-chip Blackhole. Each text encoder is placed → encoded → evicted
before the transformer is placed, and the VAE is placed lazily, so peak DRAM ≈
max(component). Per-component times (CLIP→te1, T5→te2, transformer→unet_steps,
VAE→vae) go into ``self._perf`` for the harness.

Built once; ``generate()`` runs twice (warmup + steady).
"""

import gc
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import FluxPipeline
from infra.utilities.torch_multichip_utils import enable_spmd
from loguru import logger
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.flux.pytorch.src.model_utils import (
    DTYPE,
    GUIDANCE_SCALE,
    HEIGHT,
    MAX_SEQUENCE_LENGTH,
    MESH_NAMES,
    MESH_SHAPES,
    REPO_ID,
    SEED,
    WIDTH,
    ClipTextEncoderWrapper,
    T5TextEncoderWrapper,
    shard_transformer_specs,
    tokenize_clip,
    tokenize_t5,
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

    @contextmanager
    def cache_context(self, *args, **kwargs):
        # FluxPipeline wraps the forward in `with transformer.cache_context(...)`
        # (diffusers CacheMixin); we don't cache, so this is a no-op.
        yield

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
    """Routes vae.decode() to TT (replicated), placed lazily after the denoise
    loop. Decode time goes into ``perf["vae"]``; raw pixels ([-1, 1]) are stashed
    on ``last_pixels`` so the harness can save them without the PIL postprocess."""

    def __init__(self, vae, mesh, perf):
        self._dev = torch_xla.device()
        self._perf = perf
        self.config = vae.config
        self.dtype = next(vae.parameters()).dtype
        self._vae = vae
        self._compiled = None
        self.last_pixels = None

    def decode(self, latents, return_dict=False):
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


class FluxConfig:
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


class FluxPipeline_TT:
    """FluxPipeline with every module on TT, transformer TP-sharded. Built once;
    generate() runs twice (warmup + steady). Raw transformer/VAE are kept so the
    TT wrappers can be rebuilt fresh each call."""

    def __init__(self, config: FluxConfig):
        self.config = config
        self._perf = {}

    def setup(self):
        # Enables SPMD + shardy annotations; required so tt-mlir gets the
        # @Sharding custom calls the presharded args need.
        enable_spmd()
        self.num_devices = xr.global_runtime_device_count()
        self.mesh_shape = MESH_SHAPES[self.num_devices]
        device_ids = np.array(range(self.num_devices))
        self.mesh = Mesh(device_ids, self.mesh_shape, MESH_NAMES)
        logger.info(
            f"FLUX.1 mesh {self.mesh_shape} (names={MESH_NAMES}) "
            f"over {self.num_devices} devices"
        )
        self.pipe = FluxPipeline.from_pretrained(REPO_ID, torch_dtype=DTYPE)
        # Keep the raw modules so wrappers can be rebuilt on each generate().
        self._raw_transformer = self.pipe.transformer
        self._raw_vae = self.pipe.vae

    def _encode(self, wrapper_cls, module, input_ids, dev):
        """Place a replicated text encoder on device, encode, evict; return CPU embeds + time."""
        wrapper = wrapper_cls(module).eval()
        module = module.to(dev)
        compiled = torch.compile(wrapper, backend="tt")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = compiled(input_ids.to(dev))
        torch_xla.sync()
        out = out.cpu().to(DTYPE)
        dt = time.perf_counter() - t0
        module = module.to("cpu")
        del compiled, wrapper
        return module, out, dt

    def generate(
        self,
        prompt: str,
        num_inference_steps: int,
        seed: Optional[int] = SEED,
    ):
        dev = torch_xla.device()
        self._perf = {
            "te1": None,
            "te2": None,
            "unet_steps": [],
            "vae": None,
            "total": None,
            "step_label": "Transformer",
        }
        t_total_start = time.perf_counter()

        # ── Stage 1: text encoders (CLIP=te1, T5=te2, replicated) → embeds, evict ──
        logger.info("[STAGE] CLIP text encoder: start")
        self.pipe.text_encoder, pooled_prompt_embeds, self._perf["te1"] = self._encode(
            ClipTextEncoderWrapper, self.pipe.text_encoder, tokenize_clip(prompt), dev
        )
        logger.info("[STAGE] CLIP text encoder: done")

        logger.info("[STAGE] T5 text encoder: start")
        self.pipe.text_encoder_2, prompt_embeds, self._perf["te2"] = self._encode(
            T5TextEncoderWrapper,
            self.pipe.text_encoder_2,
            tokenize_t5(prompt, max_sequence_length=MAX_SEQUENCE_LENGTH),
            dev,
        )
        gc.collect()
        torch_xla.sync()
        logger.info("[STAGE] T5 text encoder: done")

        # ── Stage 2: transformer (sharded) + VAE (replicated, lazy) → image ─────
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
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=self.config.height,
            width=self.config.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=GUIDANCE_SCALE,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            generator=generator,
        )
        logger.info("[STAGE] Transformer + VAE: done")

        self._perf["total"] = time.perf_counter() - t_total_start
        # Raw VAE pixels in [-1, 1], shape (1, 3, H, W) — the harness's
        # save_image() expects this range, so return it instead of the PIL.
        return vae_wrapper.last_pixels
