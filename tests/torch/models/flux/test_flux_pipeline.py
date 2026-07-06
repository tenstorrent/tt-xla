# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1-dev — full text-to-image pipeline e2e test (all modules on TT, multichip).

The diffusers ``FluxPipeline`` orchestrates the run (tokenizers, scheduler and
latent bookkeeping stay on CPU) at the source geometry/sampling params
(1024x1024, 50 steps, guidance 3.5, seq-512). Every compute module runs on
Tenstorrent via ``torch.compile(backend="tt")``:

  - CLIP text encoder (CLIPTextModel)      → replicated
  - T5   text encoder (T5EncoderModel)     → replicated
  - transformer (FluxTransformer2DModel)   → tensor-parallel sharded (model axis)
  - VAE decoder (AutoencoderKL)            → replicated

Runs on a 4-chip Blackhole (lb-blackhole): the transformer is SPMD-sharded with
the same shard specs as the component test (loader.shard_transformer_specs), so
its ~24GB bf16 weights split ~4-way.

Memory strategy (peak ≈ max(component)): the CLIP and T5 encoders are each
placed → used → evicted before the transformer is placed, and the VAE is placed
lazily at first decode (after the denoise loop).
"""

import gc
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import FluxPipeline
from infra import RunMode
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger
from PIL import Image
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.flux.pytorch.src.model_utils import (
    DTYPE,
    GUIDANCE_SCALE,
    HEIGHT,
    MAX_SEQUENCE_LENGTH,
    MESH_NAMES,
    MESH_SHAPES,
    PROMPT,
    REPO_ID,
    SEED,
    WIDTH,
    ClipTextEncoderWrapper,
    T5TextEncoderWrapper,
    shard_transformer_specs,
    tokenize_clip,
    tokenize_t5,
)

NUM_INFERENCE_STEPS = 50


class _DeviceDenoiser:
    """Routes FluxPipeline's transformer calls to the TP-sharded model on TT."""

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

    @contextmanager
    def cache_context(self, *args, **kwargs):
        # FluxPipeline wraps the forward in `with transformer.cache_context(...)`
        # (diffusers CacheMixin); we don't cache, so this is a no-op.
        yield

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
    """Routes FluxPipeline's vae.decode() to TT (replicated), placed lazily."""

    def __init__(self, vae, mesh):
        self._dev = torch_xla.device()
        self.config = vae.config
        self.dtype = next(vae.parameters()).dtype
        self._vae = vae
        self._compiled = None

    def decode(self, latents, return_dict=False):
        # Lazy device placement: keep the VAE off-device during the denoise loop
        # so it does not inflate the denoiser's peak DRAM; place it only now.
        if self._compiled is None:
            vae = self._vae.to(self._dev)
            self._compiled = torch.compile(
                lambda z: vae.decode(z, return_dict=False)[0], backend="tt"
            )
        out = self._compiled(latents.to(self._dev))
        torch_xla.sync()
        image = out.cpu() if torch.is_tensor(out) else out
        return (image,)


class FluxTTPipeline:
    """diffusers FluxPipeline with every module on TT, transformer TP-sharded."""

    def __init__(self, height: int = HEIGHT, width: int = WIDTH):
        self.height = height
        self.width = width

    def setup(self):
        # Enables SPMD + shardy annotations; required so the StableHLO handed to
        # tt-mlir carries the @Sharding custom calls the presharded args need.
        enable_spmd()
        self.num_devices = xr.global_runtime_device_count()
        # "model" axis is degree 4 (the shard specs' TP degree); extra devices
        # go to the replicated "batch" axis.
        self.mesh = get_mesh(MESH_SHAPES[self.num_devices], MESH_NAMES)
        self.pipe = FluxPipeline.from_pretrained(REPO_ID, torch_dtype=DTYPE)

    def _encode(self, wrapper_cls, module, input_ids, dev):
        """Place a replicated text encoder on device, encode, evict; return CPU embeds."""
        wrapper = wrapper_cls(module).eval()
        module = module.to(dev)
        compiled = torch.compile(wrapper, backend="tt")
        with torch.no_grad():
            out = compiled(input_ids.to(dev))
        torch_xla.sync()
        out = out.cpu().to(DTYPE)
        module = module.to("cpu")
        del compiled, wrapper
        return module, out

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        seed: Optional[int] = SEED,
    ):
        dev = torch_xla.device()

        # ── Stage 1: text encoders (CLIP + T5, replicated) → embeds, then evict ──
        logger.info("[STAGE] CLIP text encoder: start")
        self.pipe.text_encoder, pooled_prompt_embeds = self._encode(
            ClipTextEncoderWrapper, self.pipe.text_encoder, tokenize_clip(prompt), dev
        )
        logger.info("[STAGE] CLIP text encoder: done")

        logger.info("[STAGE] T5 text encoder: start")
        self.pipe.text_encoder_2, prompt_embeds = self._encode(
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
        self.pipe.transformer = _DeviceDenoiser(self.pipe.transformer, self.mesh)
        self.pipe.vae = _DeviceVAEDecoder(self.pipe.vae, self.mesh)

        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        result = self.pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=self.height,
            width=self.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=GUIDANCE_SCALE,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            generator=generator,
        )
        logger.info("[STAGE] Transformer + VAE: done")
        return result.images[0]


@pytest.mark.tensor_parallel
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.lb_blackhole
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="Flux1_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_flux_pipeline():
    """Run the full FLUX.1-dev pipeline with all modules on TT (sharded, multichip)."""
    xr.set_device_type("TT")
    torch.manual_seed(SEED)

    output_path = "flux1_pipeline_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    pipeline = FluxTTPipeline()
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
