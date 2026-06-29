# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
SRPO (tencent/SRPO) text-to-image example.

SRPO is a FLUX.1-dev fine-tune that ships only the MMDiT transformer (the
denoiser). The tt-forge-models loader builds that transformer from the
FLUX.1-dev config and injects the SRPO weights; the remaining FLUX components
(T5 + CLIP text encoders, VAE, tokenizers, FlowMatch scheduler) are the standard
FLUX.1-dev parts.

This example runs the realistic composite pipeline: the heavy denoiser runs on
the TT device (compiled with ``torch.compile(backend="tt")``) while the text
encoders, VAE and scheduler stay in host Python. Only the transformer's forward
is intercepted to move its tensors on/off device, so the stock ``FluxPipeline``
drives the generation loop unmodified.

Note: FLUX.1-dev is gated on the Hugging Face Hub — an ``HF_TOKEN`` with access
is required to fetch the non-transformer components.
"""

import argparse
import time

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers import FluxPipeline

from third_party.tt_forge_models.srpo.pytorch import ModelLoader, ModelVariant

# FLUX.1-dev supplies every non-transformer component (VAE, CLIP, T5, tokenizers,
# scheduler); the loader injects the SRPO transformer into this same pipeline.
FLUX_BASE = "black-forest-labs/FLUX.1-dev"


class OnDeviceDenoiser:
    """Run the FLUX/SRPO denoiser on the TT device inside a host FluxPipeline.

    The wrapped transformer keeps all of its attributes (``config``, ``dtype``,
    ``cache_context``, ...) so the stock ``FluxPipeline`` is unaware of the swap.
    Each call moves the (CPU) tensor inputs to the device, runs the compiled
    forward, and moves the prediction back to the host for the scheduler step.
    """

    def __init__(self, transformer, device):
        # Weights live on device; torch.compile lowers the forward to the TT
        # backend. The single Blackhole chip fits the bf16 denoiser at native
        # 1024 this way (a raw xla_device placement OOMs).
        transformer = transformer.to(device)
        object.__setattr__(self, "_inner", transformer)
        object.__setattr__(self, "_compiled", torch.compile(transformer, backend="tt"))
        object.__setattr__(self, "_device", device)

    def __getattr__(self, name):
        # Delegate every attribute the pipeline reads to the real transformer.
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __call__(self, **kwargs):
        moved = {
            k: (v.to(self._device) if torch.is_tensor(v) else v)
            for k, v in kwargs.items()
        }
        out = self._compiled(**moved)
        # FluxPipeline calls the transformer with return_dict=False -> tuple.
        if isinstance(out, tuple):
            return tuple(o.cpu() if torch.is_tensor(o) else o for o in out)
        return out.cpu()


def load_pipeline():
    """Build a FluxPipeline with the SRPO denoiser running on the TT device."""
    device = torch_xla.device()

    loader = ModelLoader(ModelVariant.BASE)
    # SRPO weights ship as fp32; the device baseline runs bf16 end-to-end (the
    # latent dtype must match the VAE), so load the denoiser directly in bf16.
    transformer = loader.load_model(dtype_override=torch.bfloat16).eval()

    pipe = FluxPipeline.from_pretrained(
        FLUX_BASE, transformer=transformer, torch_dtype=torch.bfloat16
    )
    # Swap the host transformer for the on-device wrapper; the rest of the
    # pipeline (text encoders, VAE, scheduler) stays in host Python.
    pipe.transformer = OnDeviceDenoiser(transformer, device)
    return pipe, loader


def generate(pipe, loader, prompt, num_inference_steps, seed=42):
    """Generate a single 1024x1024 image with the SRPO pipeline."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    start = time.time()
    image = pipe(
        prompt=prompt,
        height=loader.SAMPLE_SIZE,
        width=loader.SAMPLE_SIZE,
        guidance_scale=loader.GUIDANCE_SCALE,
        num_inference_steps=num_inference_steps,
        max_sequence_length=loader.MAX_SEQUENCE_LENGTH,
        generator=generator,
    ).images[0]
    print(f"Generation time: {time.time() - start:.1f} s")
    return image


def post_process_output(image, prompt, output_path="srpo_output.png"):
    """Save the generated image and print a human-readable summary."""
    image.save(output_path)
    print(f'Prompt: "{prompt}"')
    print(f"Generated image: {image.width}x{image.height} -> saved to {output_path}")
    return output_path


def test_srpo():
    """Smoke test: the SRPO pipeline produces a valid 1024x1024 image on device."""
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 0})

    pipe, loader = load_pipeline()
    # One step keeps CI cheap; native resolution is kept (geometry is faithful).
    image = generate(pipe, loader, "a cat", num_inference_steps=1)

    assert image.size == (loader.SAMPLE_SIZE, loader.SAMPLE_SIZE), (
        f"expected {loader.SAMPLE_SIZE}x{loader.SAMPLE_SIZE}, got {image.size}"
    )
    extrema = image.convert("L").getextrema()
    assert extrema[0] != extrema[1], "image is a flat (blank) frame"
    print("SRPO produced a valid 1024x1024 image.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="a majestic lion sitting on a rock at sunset, photorealistic",
    )
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="srpo_output.png")
    args = parser.parse_args()

    xr.set_device_type("TT")
    # Blackhole requires optimization_level 0 (>=1 aborts on this part); this
    # also matches the compiler options the bringup gate compiled the denoiser
    # with, keeping the first (cold) compile fast.
    torch_xla.set_custom_compile_options({"optimization_level": 0})

    pipe, loader = load_pipeline()
    image = generate(pipe, loader, args.prompt, args.num_inference_steps, args.seed)
    post_process_output(image, args.prompt, args.output_path)
