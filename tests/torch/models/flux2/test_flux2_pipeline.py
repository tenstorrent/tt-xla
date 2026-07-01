# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev — full text-to-image pipeline e2e test (all modules on TT).

The standard diffusers ``Flux2Pipeline`` orchestrates the run (tokenizer +
scheduler stay on CPU), but every compute module runs on Tenstorrent, compiled
with ``torch.compile(backend="tt")`` and tensor-parallel sharded via the same
SPMD shard specs as the component tests:

  - text encoder (Mistral3, ~24B)  → sharded
  - transformer  (Flux2, ~32B)     → sharded
  - VAE decoder  (~84M)            → replicated

Memory strategy (peak ≈ max(component) rather than the sum):
  * Stage 1 places the text encoder on device, encodes the prompt, then evicts it.
  * Stage 2 routes the pipeline's transformer/VAE calls through compiled wrappers
    that move inputs to device and return CPU tensors each call, so the denoise
    loop keeps only one step's activations resident. The VAE is placed lazily at
    first decode (after the denoise loop) so it never inflates the denoise peak.
"""

from pathlib import Path
from typing import Optional

import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import Flux2Pipeline
from infra import RunMode
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger
from PIL import Image
from utils import BringupStatus, Category, ModelGroup

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
        torch_xla.sync()
        if isinstance(out, (tuple, list)):
            return type(out)(o.cpu() if torch.is_tensor(o) else o for o in out)
        return out.cpu()


class _DeviceVAEDecoder:
    """Routes Flux2Pipeline's vae.decode() to TT (replicated), placed lazily.

    The pipeline reads ``vae.bn`` / ``vae.config`` / ``vae.dtype`` for the host-side
    batch-norm denorm, then calls ``vae.decode(latents, return_dict=False)[0]``.
    """

    def __init__(self, vae, mesh):
        self._dev = torch_xla.device()
        self.config = vae.config
        self.dtype = next(vae.parameters()).dtype
        self.bn = vae.bn  # stays on CPU; pipeline reads it host-side for denorm
        self._vae = vae
        self._compiled = None

    def decode(self, latents, return_dict=False):
        # Lazy device placement: keep the VAE off-device during the denoise loop so
        # it does not inflate the denoiser's peak DRAM; place it only now.
        if self._compiled is None:
            vae = self._vae.to(self._dev)
            self._compiled = torch.compile(
                lambda z: vae.decode(z, return_dict=False)[0], backend="tt"
            )
        out = self._compiled(latents.to(self._dev))
        torch_xla.sync()
        image = out.cpu() if torch.is_tensor(out) else out
        return (image,)


class Flux2TTPipeline:
    """diffusers Flux2Pipeline with every module on TT, tensor-parallel sharded."""

    def __init__(self, height: int = HEIGHT, width: int = WIDTH):
        self.height = height
        self.width = width

    def setup(self):
        enable_spmd()
        self.num_devices = xr.global_runtime_device_count()
        # Mesh from device count: "model" axis is always degree 4 (the shard
        # specs' contraction-parallel degree), extra devices go to "batch".
        self.mesh = get_mesh(MESH_SHAPES[self.num_devices], MESH_NAMES)
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
        te_specs = shard_text_encoder_specs(text_encoder)
        assert te_specs, "text-encoder shard spec is empty — descent failed (would OOM)"
        for tensor, spec in te_specs.items():
            xs.mark_sharding(tensor, self.mesh, spec)
        te_compiled = torch.compile(encoder_wrapper, backend="tt")

        with torch.no_grad():
            prompt_embeds = te_compiled(input_ids.to(dev), attention_mask.to(dev))
        torch_xla.sync()
        prompt_embeds = prompt_embeds.cpu()

        # Free the 24B encoder from device before placing the 32B denoiser.
        self.pipe.text_encoder = text_encoder.to("cpu")
        del te_compiled, encoder_wrapper
        import gc

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


@pytest.mark.tensor_parallel
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.qb2_blackhole
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="Flux2_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_flux2_pipeline():
    """Run the full FLUX.2-dev pipeline with all modules on TT (sharded)."""
    xr.set_device_type("TT")
    torch.manual_seed(SEED)

    output_path = "flux2_pipeline_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    pipeline = Flux2TTPipeline()
    pipeline.setup()
    image = pipeline.generate(
        PROMPT, num_inference_steps=NUM_INFERENCE_STEPS, seed=SEED
    )
    image.save(output_path)

    assert output_file.exists(), f"Output image {output_path} was not created"
    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"
    logger.info(f"Output image saved to {output_path} ({width}x{height})")
