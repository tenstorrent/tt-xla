# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev — text-to-image demo on Tenstorrent hardware (multichip).

The diffusers ``Flux2Pipeline`` orchestrates the run (tokenizer + scheduler stay
on CPU), but every compute module runs on Tenstorrent, compiled with
``torch.compile(backend="tt")`` and tensor-parallel sharded over the mesh's
"model" axis:

  - text encoder (Mistral3, ~24B)  → sharded
  - transformer  (Flux2, ~32B)     → sharded
  - VAE decoder  (~84M)            → replicated

Memory strategy (peak DRAM ≈ max(component) rather than the sum): the text
encoder is evicted before the transformer is placed, and the VAE is placed
lazily at first decode, so only one big module is resident on TT DRAM at a time.

Uses a degree-4 tensor-parallel mesh (1, 4) — 4 chips. ``TT_VISIBLE_DEVICES``
is pinned to the first four chips so the demo runs the same way on a 4-chip
(qb2) or 8-chip (lb) blackhole.

Run:
    python examples/pytorch/flux2.py
"""

import os

# Pin to 4 chips (mesh (1, 4)); must be set before the PJRT runtime enumerates
# devices, hence before importing torch_xla.
os.environ.setdefault("TT_VISIBLE_DEVICES", "0,1,2,3")

import gc
from pathlib import Path
from typing import Optional

import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import Flux2Pipeline
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger

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

NUM_INFERENCE_STEPS = 50


class _DeviceDenoiser:
    """Routes Flux2Pipeline's transformer calls to the TP-sharded model on TT."""

    def __init__(self, transformer, mesh):
        self._dev = torch_xla.device()
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
        out = self._compiled(**moved)
        # .cpu() is the sync point: it forces the pending graph to execute and only
        # returns once the result is on host, so no explicit sync is needed.
        if isinstance(out, (tuple, list)):
            return type(out)(o.cpu() if torch.is_tensor(o) else o for o in out)
        return out.cpu()


class _DeviceVAEDecoder:
    """Routes Flux2Pipeline's vae.decode() to TT (replicated), placed lazily."""

    def __init__(self, vae, mesh):
        self._dev = torch_xla.device()
        self.config = vae.config
        self.dtype = next(vae.parameters()).dtype
        self.bn = vae.bn  # stays on CPU; pipeline reads it host-side for denorm
        self._vae = vae
        self._compiled = None

    def decode(self, latents, return_dict=False):
        # Lazy device placement so the VAE does not inflate the denoiser's peak DRAM.
        if self._compiled is None:
            vae = self._vae.to(self._dev)
            self._compiled = torch.compile(
                lambda z: vae.decode(z, return_dict=False)[0], backend="tt"
            )
        out = self._compiled(latents.to(self._dev))
        image = out.cpu()
        return (image,)


class Flux2TTPipeline:
    """diffusers Flux2Pipeline with every module on TT, tensor-parallel sharded."""

    def __init__(self, height: int = HEIGHT, width: int = WIDTH):
        self.height = height
        self.width = width

    def setup(self):
        enable_spmd()
        self.num_devices = xr.global_runtime_device_count()
        self.mesh = get_mesh(MESH_SHAPES[self.num_devices], MESH_NAMES)
        logger.info(
            f"FLUX.2 mesh {MESH_SHAPES[self.num_devices]} (names={MESH_NAMES}) "
            f"over {self.num_devices} devices"
        )
        self.pipe = Flux2Pipeline.from_pretrained(REPO_ID, torch_dtype=DTYPE)

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        seed: Optional[int] = SEED,
    ):
        dev = torch_xla.device()

        # ── Stage 1: text encoder (sharded, compiled) → prompt embeds, evict ──
        logger.info("[STAGE] Text encoder: start")
        text_encoder = self.pipe.text_encoder
        encoder_wrapper = Mistral3TextEncoderWrapper(text_encoder).eval()
        input_ids, attention_mask = tokenize_prompt(prompt)

        text_encoder = text_encoder.to(dev)
        if hasattr(text_encoder, "tie_weights"):
            text_encoder.tie_weights()
        for tensor, spec in shard_text_encoder_specs(text_encoder).items():
            xs.mark_sharding(tensor, self.mesh, spec)
        te_compiled = torch.compile(encoder_wrapper, backend="tt")

        with torch.no_grad():
            prompt_embeds = te_compiled(input_ids.to(dev), attention_mask.to(dev))
        prompt_embeds = prompt_embeds.cpu()

        # Free the 24B encoder from device before placing the 32B denoiser.
        self.pipe.text_encoder = text_encoder.to("cpu")
        del te_compiled, encoder_wrapper
        gc.collect()
        torch_xla.sync()
        logger.info("[STAGE] Text encoder: done")

        # ── Stage 2: denoiser (sharded) + VAE (replicated, lazy) → image ─────
        logger.info("[STAGE] Transformer + VAE: start")
        self.pipe.transformer = _DeviceDenoiser(self.pipe.transformer, self.mesh)
        self.pipe.vae = _DeviceVAEDecoder(self.pipe.vae, self.mesh)

        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        result = self.pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            height=self.height,
            width=self.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        )
        logger.info("[STAGE] Transformer + VAE: done")
        return result.images[0]


def run_flux2(
    output_path: str = "flux2_output.png",
    num_inference_steps: int = NUM_INFERENCE_STEPS,
):
    """Run the FLUX.2-dev pipeline end-to-end on TT and save the output image."""
    pipeline = Flux2TTPipeline()
    pipeline.setup()
    image = pipeline.generate(
        PROMPT, num_inference_steps=num_inference_steps, seed=SEED
    )
    image.save(output_path)
    return output_path


if __name__ == "__main__":
    xr.set_device_type("TT")
    torch.manual_seed(SEED)
    output_path = "flux2_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()
    run_flux2(output_path=output_path)
    logger.info(f"Output image saved to {output_path}")
