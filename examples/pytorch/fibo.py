# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable FIBO (briaai/FIBO) text-to-image example on Tenstorrent.

FIBO is BRIA AI's 8B-parameter DiT flow-matching text-to-image model (SmolLM3-3B
text encoder, Wan 2.2 VAE). The 8B DiT does not fit in a single Wormhole/Blackhole
chip's DRAM, so — unlike the single-device SD1.5 / SD3 demos — the heavy net runs
**tensor-parallel across the multi-chip mesh** (Megatron-1D over a ``(None, "model")``
mesh); the text encoder, scheduler and VAE stay on CPU.

The reusable pipeline lives in ``tt_forge_models`` (``fibo/pytorch/pipeline.py``)
and is shared with the bringup/benchmark path, so this script is just a thin
runnable demo that drives it — the implementation isn't duplicated in ``examples/``.

Batch-size-one configuration: the loader pins ``guidance_scale = 1.0`` (classifier-
free guidance off), so ``BriaFiboPipeline`` never doubles the transformer batch and
the DiT runs at batch=1. ``generate()`` reads the scale from the loader, so this
demo does not pass one.
"""

from pathlib import Path

import torch
import torch_xla
import torch_xla.runtime as xr
from PIL import Image

from third_party.tt_forge_models.fibo.pytorch.pipeline import FiboConfig, FiboPipeline

# FIBO is trained on structured (JSON) captions; the pipeline tokenizes via
# SmolLM3 either way. This mirrors the model card's Generate example format.
PROMPT = (
    '{"subject":"a red panda astronaut floating inside a glass space station",'
    '"style_medium":"photograph","camera":"35mm, shallow depth of field",'
    '"lighting":"warm rim light against deep-space blue"}'
)
NEGATIVE_PROMPT = None
NUM_INFERENCE_STEPS = 50  # FIBO model-card default
SEED = 42
OUTPUT_PATH = "fibo_output.png"

# qb2-blackhole aborts compilation at optimization_level >= 1 (OpModel mock-device
# grid mismatch vs the system descriptor), so pin level 0 — the same option the
# FIBO bringup/perf baseline uses for its first compile.
COMPILE_OPTIONS = {"optimization_level": 0}


def save_image(image: torch.Tensor, filepath: str = OUTPUT_PATH):
    """Save the pipeline output image.

    ``FiboPipeline.generate`` returns a ``(1, 3, H, W)`` float tensor already in
    ``[0, 1]`` (diffusers ``output_type="pt"``), so no ``/2 + 0.5`` denorm is
    needed — just clamp, scale to uint8 and write.
    """
    image = (torch.clamp(image, 0.0, 1.0) * 255.0).round().to(dtype=torch.uint8)
    image_np = image.cpu().squeeze().numpy()
    assert image_np.ndim == 3, "Image must be 3D"
    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)
    Image.fromarray(image_np).save(filepath)


def run_fibo(
    output_path: str = OUTPUT_PATH,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
) -> torch.Tensor:
    """Build FIBO via the loader, run one generation on the mesh, save the image."""
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options(COMPILE_OPTIONS)

    pipeline = FiboPipeline(config=FiboConfig())
    pipeline.setup()

    image = pipeline.generate(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=num_inference_steps,
        seed=SEED,
    )

    save_image(image, output_path)
    return image


def post_process_output(image: torch.Tensor, output_path: str = OUTPUT_PATH):
    """Print a human-readable summary of the generated image."""
    _, _, height, width = image.shape
    print(f"Prompt: {PROMPT}")
    print(f"Generated a {width}x{height} image (tensor {tuple(image.shape)}).")
    print(f"Saved output image to {output_path}")


def test_fibo():
    """FIBO generates a valid 1024x1024 image on the Tenstorrent mesh."""
    output_path = "test_fibo_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    try:
        image = run_fibo(output_path=output_path)

        assert torch.isfinite(image).all(), "Image tensor contains non-finite values"
        assert output_file.exists(), f"Output image {output_path} was not created"

        with Image.open(output_path) as img:
            width, height = img.size
        assert (width, height) == (1024, 1024), f"Expected 1024x1024, got {width}x{height}"

        print(f"Output image created with resolution {width}x{height}")

    finally:
        if output_file.exists():
            output_file.unlink()
            print(f"Cleaned up {output_path}")


if __name__ == "__main__":
    image = run_fibo()
    post_process_output(image, OUTPUT_PATH)
