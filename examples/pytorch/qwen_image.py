# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image — text-to-image demo on Tenstorrent hardware.

The diffusers QwenImagePipeline orchestrates (tokenizer + scheduler stay on CPU);
every compute module runs on Tenstorrent: the Qwen2.5-VL text encoder and the
QwenImage MMDiT transformer are both tensor-parallel sharded across the mesh
model axis, and the VAE decoder is replicated. Each module is placed and evicted
in turn so at most one heavy component is resident on TT DRAM at a time.

Run (on a 4-chip Blackhole mesh):
    python examples/pytorch/qwen_image.py
"""

import gc
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import QwenImagePipeline
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger

from third_party.tt_forge_models.qwen_image.pytorch.src.model_utils import (
    DTYPE,
    HEIGHT,
    MESH_NAMES,
    MESH_SHAPES,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    POSITIVE_MAGIC,
    PROMPT,
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
    """Transformer on TT (tensor-parallel sharded)."""

    def __init__(self, transformer, mesh):
        self._dev = torch_xla.device()
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
        (sample,) = self._compiled(**moved)
        return (sample.cpu(),)


class _DeviceVAEDecoder:
    """VAE decode on TT (replicated)."""

    def __init__(self, vae):
        self._dev = torch_xla.device()
        self.config = vae.config
        self.dtype = next(vae.parameters()).dtype
        self.temperal_downsample = vae.temperal_downsample
        self._vae = vae
        self._compiled = None

    def decode(self, latents, return_dict=False):
        # Lazy device placement: keep the VAE off-device during the denoise loop.
        if self._compiled is None:
            vae = self._vae.to(self._dev)
            self._compiled = torch.compile(
                lambda z: vae.decode(z, return_dict=False)[0], backend="tt"
            )
        return (self._compiled(latents.to(self._dev)).cpu(),)


class QwenImagePipelineTT:
    """QwenImagePipeline with every compute module on TT."""

    def setup(self):
        enable_spmd()
        self.mesh = get_mesh(
            MESH_SHAPES[xr.global_runtime_device_count()], MESH_NAMES
        )
        self.pipe = QwenImagePipeline.from_pretrained(REPO_ID, torch_dtype=DTYPE)

    def generate(self, prompt, num_inference_steps=NUM_INFERENCE_STEPS, seed=SEED):
        # Stage 1: text encoder → prompt embeds (host-side masked extraction runs
        # on CPU), then evict before the transformer.
        logger.info("[STAGE] Text encoder: start")
        text_encoder = self.pipe.text_encoder
        te_wrapper = _DeviceTextEncoder(text_encoder, self.mesh)
        self.pipe.text_encoder = te_wrapper
        cpu = torch.device("cpu")
        prompt_embeds, prompt_embeds_mask = self.pipe.encode_prompt(
            prompt=prompt + POSITIVE_MAGIC,
            device=cpu,
            num_images_per_prompt=1,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )
        negative_prompt_embeds, negative_prompt_embeds_mask = self.pipe.encode_prompt(
            prompt=NEGATIVE_PROMPT,
            device=cpu,
            num_images_per_prompt=1,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )
        # Evict the text encoder before placing the transformer (flux2 pattern).
        self.pipe.text_encoder = text_encoder.to("cpu")
        del te_wrapper
        gc.collect()
        torch_xla.sync()
        logger.info("[STAGE] Text encoder: done")

        # Stage 2: transformer (sharded) + VAE (replicated) → image.
        logger.info("[STAGE] Transformer + VAE: start")
        self.pipe.transformer = _DeviceDenoiser(self.pipe.transformer, self.mesh)
        self.pipe.vae = _DeviceVAEDecoder(self.pipe.vae)

        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        result = self.pipe(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=TRUE_CFG_SCALE,
            generator=generator,
        )
        logger.info("[STAGE] Transformer + VAE: done")
        return result.images[0]


def run_qwen_image(
    output_path: str = "qwen_image_output.png",
    num_inference_steps: int = NUM_INFERENCE_STEPS,
):
    """Run the Qwen-Image pipeline end-to-end on TT and save the output image."""
    pipeline = QwenImagePipelineTT()
    pipeline.setup()
    image = pipeline.generate(
        PROMPT, num_inference_steps=num_inference_steps, seed=SEED
    )
    image.save(output_path)
    return output_path


if __name__ == "__main__":
    xr.set_device_type("TT")
    torch.manual_seed(SEED)
    output_path = "qwen_image_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()
    run_qwen_image(output_path=output_path)
    logger.info(f"Output image saved to {output_path}")
