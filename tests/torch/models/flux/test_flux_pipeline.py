# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1-dev — full text-to-image pipeline e2e test (all modules on TT).

The standard diffusers ``FluxPipeline`` orchestrates the run (tokenizers +
scheduler stay on CPU, and so does all the latent-packing / timestep / guidance
bookkeeping), so the geometry and sampling params are exactly the source
inference pipeline's (1024x1024, 50 steps, guidance 3.5, seq-512). Every compute
module runs on Tenstorrent, compiled with ``torch.compile(backend="tt")``:

  - CLIP text encoder (CLIPTextModel,  ~0.12B) → pooled embedding
  - T5  text encoder (T5EncoderModel,  ~4.7B)  → sequence embedding
  - transformer      (FluxTransformer2DModel,  ~12B) → denoise loop
  - VAE decoder      (AutoencoderKL,   ~84M)   → image

Single-chip Blackhole memory strategy (peak ≈ max(component), not the sum):
  * Stage 1 places each text encoder on device → encodes the prompt → evicts it,
    so the two encoders never coexist on device with the transformer.
  * Stage 2 routes the pipeline's transformer/VAE calls through compiled wrappers
    that move inputs to device and return CPU tensors each call, so the denoise
    loop keeps only one step's activations resident. The transformer's weights
    are converted to bfp8 (+ dram-space-saving) so they fit single-chip — bf16
    weights (~23.8GB) + activations OOM by ~132MB (issue #5251); the VAE is placed
    lazily at first decode (after the denoise loop).

The T5 encoder runs on device with its known bf16 PCC drift (~0.9575, issue
#5250); this e2e exercises whether the assembled image stays coherent regardless.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers import FluxPipeline
from infra import RunMode
from loguru import logger
from PIL import Image
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.flux.pytorch.src.model_utils import (
    DTYPE,
    GUIDANCE_SCALE,
    HEIGHT,
    MAX_SEQUENCE_LENGTH,
    PROMPT,
    REPO_ID,
    SEED,
    WIDTH,
    ClipTextEncoderWrapper,
    T5TextEncoderWrapper,
    tokenize_clip,
    tokenize_t5,
)

NUM_INFERENCE_STEPS = 50

# The transformer's bf16 weights (~23.8GB) + denoise activations exceed single-
# chip DRAM (OOMs by ~132MB even with dram-space-saving), so convert its
# matmul/linear weights to bfp8 (~12GB) and keep dram-space-saving (issue #5251).
_TRANSFORMER_OPTIONS = {
    "experimental-enable-dram-space-saving-optimization": "true",
    "experimental_weight_dtype": "bfp_bf8",
}


class _DeviceDenoiser:
    """Routes FluxPipeline's transformer calls to the compiled model on TT.

    Compiled with bfp8 weights + DRAM space-saving so the transformer fits
    single-chip; exposes the ``config`` / ``dtype`` the pipeline reads
    (guidance_embeds, in_channels).
    """

    def __init__(self, transformer):
        self._dev = torch_xla.device()
        self.config = transformer.config
        self.dtype = next(transformer.parameters()).dtype

        transformer = transformer.to(self._dev)
        torch_xla.set_custom_compile_options(_TRANSFORMER_OPTIONS)
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
    """Routes FluxPipeline's vae.decode() to TT, placed lazily after denoise.

    The pipeline unpacks + denorms latents host-side (reading ``vae.config``),
    then calls ``vae.decode(latents, return_dict=False)[0]``.
    """

    def __init__(self, vae):
        self._dev = torch_xla.device()
        self.config = vae.config
        self.dtype = next(vae.parameters()).dtype
        self._vae = vae
        self._compiled = None

    def decode(self, latents, return_dict=False):
        # Lazy device placement: keep the VAE off-device during the denoise loop
        # so it does not inflate the denoiser's peak DRAM; place it only now.
        if self._compiled is None:
            # Reset to default opt (drop DRAM space-saving): the VAE passes on
            # default options and is small enough not to need it.
            torch_xla.set_custom_compile_options({})
            vae = self._vae.to(self._dev)
            self._compiled = torch.compile(
                lambda z: vae.decode(z, return_dict=False)[0], backend="tt"
            )
        out = self._compiled(latents.to(self._dev))
        torch_xla.sync()
        image = out.cpu() if torch.is_tensor(out) else out
        return (image,)


class FluxTTPipeline:
    """diffusers FluxPipeline with every compute module on TT (single chip)."""

    def __init__(self, height: int = HEIGHT, width: int = WIDTH):
        self.height = height
        self.width = width

    def setup(self):
        self.pipe = FluxPipeline.from_pretrained(REPO_ID, torch_dtype=DTYPE)

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        seed: Optional[int] = SEED,
    ):
        dev = torch_xla.device()

        # ── Stage 1: text encoders (CLIP + T5) → embeds, then evict ──────────
        logger.info("[STAGE] CLIP text encoder: start")
        clip = ClipTextEncoderWrapper(self.pipe.text_encoder).eval()
        clip_ids = tokenize_clip(prompt)
        self.pipe.text_encoder = self.pipe.text_encoder.to(dev)
        clip_compiled = torch.compile(clip, backend="tt")
        with torch.no_grad():
            pooled_prompt_embeds = clip_compiled(clip_ids.to(dev))
        torch_xla.sync()
        pooled_prompt_embeds = pooled_prompt_embeds.cpu().to(DTYPE)
        self.pipe.text_encoder = self.pipe.text_encoder.to("cpu")
        del clip_compiled, clip
        logger.info("[STAGE] CLIP text encoder: done")

        logger.info("[STAGE] T5 text encoder: start")
        t5 = T5TextEncoderWrapper(self.pipe.text_encoder_2).eval()
        t5_ids = tokenize_t5(prompt, max_sequence_length=MAX_SEQUENCE_LENGTH)
        self.pipe.text_encoder_2 = self.pipe.text_encoder_2.to(dev)
        t5_compiled = torch.compile(t5, backend="tt")
        with torch.no_grad():
            prompt_embeds = t5_compiled(t5_ids.to(dev))
        torch_xla.sync()
        prompt_embeds = prompt_embeds.cpu().to(DTYPE)
        self.pipe.text_encoder_2 = self.pipe.text_encoder_2.to("cpu")
        del t5_compiled, t5

        import gc

        gc.collect()
        torch_xla.sync()
        logger.info("[STAGE] T5 text encoder: done")

        # ── Stage 2: transformer (denoise) + VAE (lazy) → image ─────────────
        logger.info("[STAGE] Transformer + VAE: start")
        self.pipe.transformer = _DeviceDenoiser(self.pipe.transformer)
        self.pipe.vae = _DeviceVAEDecoder(self.pipe.vae)

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


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="Flux1_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_flux_pipeline():
    """Run the full FLUX.1-dev pipeline with all modules on TT (single chip)."""
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
