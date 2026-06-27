# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
SRPO (Tencent, arXiv:2509.06942) text-to-image generation example.

SRPO is a fine-tune of the FLUX.1-dev MM-DiT transformer (the denoiser); the
published checkpoint ships *only* the transformer weights and reuses the rest of
the FLUX.1-dev pipeline (CLIP + T5 text encoders, VAE, scheduler) for inference.

This example mirrors ``sdxl-pipeline.py``: it runs a real text-to-image
diffusion pipeline, with the expensive denoiser (the SRPO ``FluxTransformer2DModel``
from the tt-forge-models loader) compiled with ``torch.compile(backend="tt")`` and
executed on the Tenstorrent device, while the host (CPU) handles text encoding,
the flow-match scheduler loop and VAE decode. The denoiser is the component that
was brought up on device; the surrounding FLUX.1-dev components are reused from
``diffusers`` unchanged.

Single-chip is the SRPO baseline on Blackhole (transformer tensor-parallelism is
blocked by a ttnn.concat L1 overflow), so this example runs the denoiser on a
single device. Images are generated at the native 1024x1024 resolution.
"""

import argparse
import time
from pathlib import Path

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers import FluxPipeline

from third_party.tt_forge_models.srpo.pytorch import ModelLoader, ModelVariant

# FLUX.1-dev / SRPO native generation settings (from the model card / loader).
NATIVE_HEIGHT = ModelLoader.NATIVE_HEIGHT  # 1024
NATIVE_WIDTH = ModelLoader.NATIVE_WIDTH  # 1024
MAX_SEQUENCE_LENGTH = ModelLoader.NATIVE_MAX_SEQUENCE_LENGTH  # 512
GUIDANCE_SCALE = ModelLoader.GUIDANCE_SCALE  # 3.5
BASE_PIPELINE = ModelLoader._BASE_PIPELINE  # black-forest-labs/FLUX.1-dev


class TTDenoiser:
    """Adapter that lets ``FluxPipeline`` drive the SRPO denoiser on the TT device.

    The diffusers pipeline calls ``self.transformer(...)`` once per denoising
    step with host (CPU) tensors. This thin wrapper casts those tensors to the
    denoiser dtype, moves them onto the Tenstorrent device, runs the compiled
    transformer, and brings the noise prediction back to the host so the
    host-side scheduler loop and VAE decode are unaffected. Only the denoiser
    runs on device; text encoders and VAE stay on CPU.
    """

    def __init__(self, denoiser: torch.nn.Module, device):
        self._denoiser = denoiser  # compiled (backend="tt") and moved to `device`
        self._device = device
        # Attributes the pipeline reads off the transformer.
        self.config = denoiser.config
        self.dtype = denoiser.dtype

    def cache_context(self, *args, **kwargs):
        # diffusers wraps each transformer call in `transformer.cache_context(...)`.
        return self._denoiser.cache_context(*args, **kwargs)

    def _to_device(self, value):
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                value = value.to(self.dtype)
            return value.to(self._device)
        return value

    def __call__(self, **kwargs):
        moved = {k: self._to_device(v) for k, v in kwargs.items()}
        output = self._denoiser(**moved)
        sample = output[0] if isinstance(output, (tuple, list)) else output.sample
        return (sample.to("cpu"),)


def build_srpo_pipeline() -> FluxPipeline:
    """Build a FLUX.1-dev pipeline with the SRPO denoiser running on the TT device."""
    device = torch_xla.device()

    # FLUX.1-dev pipeline minus the denoiser (CLIP + T5 + VAE + scheduler on host).
    # transformer=None skips loading the stock FLUX transformer; we plug in SRPO.
    pipe = FluxPipeline.from_pretrained(
        BASE_PIPELINE,
        transformer=None,
        torch_dtype=torch.bfloat16,
    )

    # SRPO denoiser (FLUX transformer architecture + SRPO weights) via the loader.
    loader = ModelLoader(ModelVariant.SRPO)
    denoiser = loader.load_model().eval()
    denoiser.compile(backend="tt")
    denoiser = denoiser.to(device)

    pipe.transformer = TTDenoiser(denoiser, device)
    return pipe


def generate(
    pipe: FluxPipeline,
    prompt: str,
    num_inference_steps: int = 4,
    seed: int = 42,
):
    """Run the diffusion pipeline for `prompt` and return a PIL image."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    start = time.time()
    image = pipe(
        prompt=prompt,
        height=NATIVE_HEIGHT,
        width=NATIVE_WIDTH,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=num_inference_steps,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        generator=generator,
    ).images[0]
    print(f"Generation time: {time.time() - start:.1f}s")
    return image


def post_process_output(image, prompt: str, output_path: str = "srpo_output.png"):
    """Save the generated image and print a human-readable result."""
    image.save(output_path)
    path = Path(output_path).resolve()
    print(f"\nSRPO text-to-image result")
    print(f"  prompt     : {prompt}")
    print(f"  resolution : {image.width}x{image.height}")
    print(f"  saved image: {path}")
    return str(path)


def run_srpo(
    prompt: str = "a majestic snow leopard on a mountain ridge at golden hour, ultra detailed",
    output_path: str = "srpo_output.png",
    num_inference_steps: int = 4,
):
    """End-to-end SRPO text-to-image generation on the TT device."""
    torch_xla.set_custom_compile_options({"optimization_level": 1})
    pipe = build_srpo_pipeline()
    image = generate(pipe, prompt, num_inference_steps=num_inference_steps)
    return image, post_process_output(image, prompt, output_path)


def test_srpo():
    """Generate one image and assert it is a valid, non-degenerate 1024x1024 result."""
    import numpy as np

    xr.set_device_type("TT")

    output_path = "test_srpo_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    try:
        # One denoising step keeps the on-device check cheap while exercising the
        # full real pipeline at native resolution.
        image, _ = run_srpo(
            prompt="a photo of a red apple on a wooden table",
            output_path=output_path,
            num_inference_steps=1,
        )

        assert output_file.exists(), f"Output image {output_path} was not created"
        assert image.width == NATIVE_WIDTH and image.height == NATIVE_HEIGHT, (
            f"Expected {NATIVE_WIDTH}x{NATIVE_HEIGHT}, got {image.width}x{image.height}"
        )

        arr = np.asarray(image, dtype=np.float32)
        assert np.isfinite(arr).all(), "Generated image contains non-finite pixels"
        assert arr.std() > 0.0, "Generated image is a constant (degenerate) image"

        print(f"Output image created with resolution {image.width}x{image.height}")
    finally:
        if output_file.exists():
            output_file.unlink()
            print(f"Cleaned up {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRPO text-to-image example")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a majestic snow leopard on a mountain ridge at golden hour, ultra detailed",
    )
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--output_path", type=str, default="srpo_output.png")
    args = parser.parse_args()

    xr.set_device_type("TT")

    run_srpo(
        prompt=args.prompt,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
    )
