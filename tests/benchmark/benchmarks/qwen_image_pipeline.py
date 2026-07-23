# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image — multichip benchmark-side pipeline for the imagegen harness.

Mirrors the nightly e2e test (tests/torch/models/qwen_image/test_pipeline.py):
the diffusers QwenImagePipeline orchestrates, the Qwen2.5-VL text encoder and
the QwenImage transformer are both tensor-parallel sharded (model axis) and the
VAE is replicated. Components are placed and evicted in turn so peak
DRAM ~= max(component). Per-component and per-step times go into ``self._perf``.
"""

import gc
import time
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import QwenImagePipeline
from infra.utilities.torch_multichip_utils import enable_spmd
from loguru import logger
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.qwen_image.pytorch.src.model_utils import (
    DTYPE,
    HEIGHT,
    MESH_NAMES,
    MESH_SHAPES,
    NEGATIVE_PROMPT,
    POSITIVE_MAGIC,
    REPO_ID,
    SEED,
    TRUE_CFG_SCALE,
    WIDTH,
    shard_text_encoder_specs,
    shard_transformer_specs,
)

MAX_SEQUENCE_LENGTH = 1024


class _DeviceTextEncoder:
    """Text encoder on TT (tensor-parallel sharded); returns hidden_states[-1]."""

    def __init__(self, text_encoder, mesh):
        self._dev = torch_xla.device()
        self.dtype = next(text_encoder.parameters()).dtype
        self.config = text_encoder.config
        text_encoder = text_encoder.to(self._dev)
        if hasattr(text_encoder, "tie_weights"):
            text_encoder.tie_weights()
        # Shard (~16.6 -> ~4 GB/chip) so it fits alongside the transformer.
        for tensor, spec in shard_text_encoder_specs(text_encoder).items():
            xs.mark_sharding(tensor, mesh, spec)
        self._compiled = torch.compile(text_encoder, backend="tt")

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=True):
        out = self._compiled(
            input_ids=input_ids.to(self._dev),
            attention_mask=(
                attention_mask.to(self._dev) if attention_mask is not None else None
            ),
            output_hidden_states=True,
        )
        return SimpleNamespace(hidden_states=(out.hidden_states[-1].cpu(),))


class _DeviceDenoiser:
    """Transformer on TT (TP-sharded); each call is one denoise step, timed."""

    def __init__(self, transformer, mesh, perf):
        self._dev = torch_xla.device()
        self._perf = perf
        self.config = transformer.config
        self.dtype = next(transformer.parameters()).dtype
        self.cache_context = transformer.cache_context

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
        # return_dict=False -> 1-tuple; .cpu() forces execution and blocks.
        (sample,) = self._compiled(**moved)
        sample = sample.cpu()
        self._perf["steps"].append(time.perf_counter() - t0)
        return (sample,)


class _DeviceVAEDecoder:
    """VAE decode on TT (replicated); timed. Stashes raw frame-0 pixels."""

    def __init__(self, vae, mesh, perf):
        self._dev = torch_xla.device()
        self._perf = perf
        self.config = vae.config
        self.dtype = next(vae.parameters()).dtype
        self.temperal_downsample = vae.temperal_downsample
        self._vae = vae
        self._compiled = None
        self.last_pixels = None

    def decode(self, latents, return_dict=False):
        # Lazy device placement: keep the VAE off-device during the denoise loop.
        if self._compiled is None:
            vae = self._vae.to(self._dev)
            self._compiled = torch.compile(
                lambda z: vae.decode(z, return_dict=False)[0], backend="tt"
            )
        t0 = time.perf_counter()
        image = self._compiled(latents.to(self._dev)).cpu()
        self._perf["components"]["vae"] = time.perf_counter() - t0
        # pipeline consumes decode(...)[0][:, :, 0]; stash that raw (B,3,H,W).
        self.last_pixels = image[:, :, 0]
        return (image,)


class QwenImageConfig:
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


class QwenImagePipeline_TT:
    """QwenImagePipeline with every module on TT, transformer TP-sharded."""

    def __init__(self, config: QwenImageConfig):
        self.config = config
        # Persistent perf dict: the cached device wrappers hold this reference, so
        # it is cleared in place (never reassigned) between passes.
        self._perf = {
            "components": {},
            "steps": [],
            "step_metric_name": "transformer_step",
            "total": None,
        }
        self._placed = False

    def setup(self):
        enable_spmd()
        self.num_devices = xr.global_runtime_device_count()
        self.mesh_shape = MESH_SHAPES[self.num_devices]
        device_ids = np.array(range(self.num_devices))
        self.mesh = Mesh(device_ids, self.mesh_shape, MESH_NAMES)
        logger.info(
            f"Qwen-Image mesh {self.mesh_shape} (names={MESH_NAMES}) "
            f"over {self.num_devices} devices"
        )
        self.pipe = QwenImagePipeline.from_pretrained(REPO_ID, torch_dtype=DTYPE)
        self._raw_transformer = self.pipe.transformer
        self._raw_vae = self.pipe.vae

    def generate(
        self, prompt: str, num_inference_steps: int, seed: Optional[int] = SEED
    ):
        self._perf["components"].clear()
        self._perf["steps"].clear()
        self._perf["total"] = None
        t_total_start = time.perf_counter()

        # Place + encode ONCE, then reuse across the harness's warmup and steady
        # passes. The stack does not free a compiled module's device memory, so
        # re-placing each pass would leave two full pipelines resident and OOM
        # the VAE decode; building once keeps residency at one pipeline.
        if not self._placed:
            # Stage 1: text encoder (sharded) → prompt embeds, then evict.
            logger.info("[STAGE] Text encoder: start")
            text_encoder = self.pipe.text_encoder
            te_wrapper = _DeviceTextEncoder(text_encoder, self.mesh)
            self.pipe.text_encoder = te_wrapper
            cpu = torch.device("cpu")
            t0 = time.perf_counter()
            pe, pem = self.pipe.encode_prompt(
                prompt=prompt + POSITIVE_MAGIC,
                device=cpu,
                num_images_per_prompt=1,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
            )
            npe, npem = self.pipe.encode_prompt(
                prompt=NEGATIVE_PROMPT,
                device=cpu,
                num_images_per_prompt=1,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
            )
            self._te_time = time.perf_counter() - t0
            self.pipe.text_encoder = text_encoder.to("cpu")
            del te_wrapper
            gc.collect()
            torch_xla.sync()
            logger.info("[STAGE] Text encoder: done")
            self._embeds = (pe, pem, npe, npem)

            # Stage 2: place transformer (sharded) + VAE (replicated, lazy) once.
            self.pipe.transformer = _DeviceDenoiser(
                self._raw_transformer, self.mesh, self._perf
            )
            self._vae_wrapper = _DeviceVAEDecoder(self._raw_vae, self.mesh, self._perf)
            self.pipe.vae = self._vae_wrapper
            self._placed = True

        self._perf["components"]["text_encoder"] = self._te_time
        (
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
        ) = self._embeds
        logger.info("[STAGE] Transformer + VAE: start")
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        self.pipe(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            height=self.config.height,
            width=self.config.width,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=TRUE_CFG_SCALE,
            generator=generator,
        )
        logger.info("[STAGE] Transformer + VAE: done")

        self._perf["total"] = time.perf_counter() - t_total_start
        # Raw VAE pixels in [-1, 1], shape (1, 3, H, W) for the harness save_image.
        return self._vae_wrapper.last_pixels
