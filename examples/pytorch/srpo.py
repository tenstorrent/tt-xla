# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
SRPO (FLUX.1-dev fine-tune) text-to-image diffusion example.

SRPO (Tencent, arXiv:2509.06942) fine-tunes the FLUX.1-dev MMDiT transformer --
the per-step "denoiser", the heavy compute of the diffusion loop. This example
runs that denoiser on the TT device while the lightweight host components of the
FLUX pipeline (CLIP + T5 text encoders, VAE, scheduler) stay on CPU, mirroring
the host-Python pipeline pattern of ``sdxl-pipeline.py`` / ``sd_v1_4_pipeline.py``.

The model + frozen FLUX.1-dev pipeline are built through the tt-forge-models
``ModelLoader`` (which loads the SRPO checkpoint into a FLUX transformer and
assembles a ``FluxPipeline`` around it). We compile that transformer with the
``tt`` backend, move it to the device, and let the diffusers pipeline drive a
native 1024x1024 generation -- every denoising step is dispatched to the device.

Note: the FLUX.1-dev components are gated on the Hugging Face Hub; set ``HF_TOKEN``
(or run ``huggingface-cli login``) before running.
"""

import argparse

import torch
import torch_xla
import torch_xla.runtime as xr
from PIL import Image

from third_party.tt_forge_models.srpo.pytorch import ModelLoader, ModelVariant

DEFAULT_PROMPT = "A photorealistic astronaut riding a horse in a futuristic city"
# SRPO generates at FLUX.1-dev's native 1024x1024 resolution (see the model card).
HEIGHT = 1024
WIDTH = 1024
# FLUX.1-dev's reference schedule is 50 steps; we use fewer here so the example
# fits the device step budget. The geometry (1024x1024, 512-token T5 context) is
# kept at the model's real configuration -- only the step count is reduced.
DEFAULT_NUM_STEPS = 8


def _move(obj, device):
    """Recursively move tensors in a (possibly nested) structure to ``device``."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move(v, device) for v in obj)
    return obj


class _DeviceDenoiser(torch.nn.Module):
    """Run the SRPO/FLUX denoiser on the TT device inside a host-CPU pipeline.

    The compiled transformer is held as a non-registered attribute so this
    wrapper exposes no device parameters: the FLUX pipeline therefore keeps its
    text encoders / VAE / scheduler on CPU (its execution device stays host),
    while each denoising call is bounced to the device and back.
    """

    def __init__(self, denoiser: torch.nn.Module, device: torch.device):
        super().__init__()
        # Bypass nn.Module registration so the device params stay hidden from the
        # pipeline's host-device detection.
        object.__setattr__(self, "_denoiser", denoiser)
        object.__setattr__(self, "_device", device)
        # diffusers reads these off the transformer during generation.
        self.config = denoiser.config
        self.dtype = denoiser.dtype

    def __getattr__(self, name):
        # Delegate anything we don't define (e.g. diffusers' cache_context) to the
        # wrapped transformer, without registering it as a submodule.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_denoiser"), name)

    def forward(self, *args, **kwargs):
        args = _move(args, self._device)
        kwargs = _move(kwargs, self._device)
        out = self._denoiser(*args, **kwargs)
        return _move(out, "cpu")


def srpo(prompt: str = DEFAULT_PROMPT, num_inference_steps: int = DEFAULT_NUM_STEPS, seed: int = 0):
    """Generate an image with the SRPO denoiser running on the TT device."""
    device = torch_xla.device()

    # Build the SRPO transformer + frozen FLUX.1-dev pipeline via the loader.
    loader = ModelLoader(ModelVariant.DEV)
    transformer = loader.load_model().eval()
    pipe = loader.pipe

    # Compile the denoiser for the tt backend and move it to the device, then
    # wrap it so the rest of the pipeline can keep running on host CPU.
    transformer.compile(backend="tt")
    transformer = transformer.to(device)
    pipe.transformer = _DeviceDenoiser(transformer, device)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            height=HEIGHT,
            width=WIDTH,
            guidance_scale=loader.guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
    return result.images[0]


def post_process_output(image: Image.Image, output_path: str = "srpo_output.png"):
    """Save the generated image and print a human-readable result."""
    image.save(output_path)
    print(f"Generated a {image.width}x{image.height} image: {output_path}")
    return output_path


def test_srpo():
    """Smoke-test that the SRPO denoiser runs on device and yields a real image."""
    import numpy as np

    xr.set_device_type("TT")

    # One denoising step keeps the test cheap while still exercising the full
    # device compile + dispatch path at native resolution.
    image = srpo(num_inference_steps=1, seed=0)

    assert image.size == (WIDTH, HEIGHT), f"expected {WIDTH}x{HEIGHT}, got {image.size}"
    arr = np.asarray(image)
    assert np.isfinite(arr).all(), "image contains non-finite pixels"
    assert arr.std() > 0, "image is flat (no content)"
    print("SRPO example produced a valid image on device.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRPO text-to-image example")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="srpo_output.png")
    args = parser.parse_args()

    xr.set_device_type("TT")

    image = srpo(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
    )
    post_process_output(image, args.output)
