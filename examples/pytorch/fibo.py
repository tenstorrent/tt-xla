# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable FIBO (briaai/FIBO) text-to-image example on Tenstorrent.

FIBO is BRIA AI's 8B-parameter DiT flow-matching text-to-image model (SmolLM3-3B
text encoder, Wan 2.2 VAE, ``BriaFiboTransformer2DModel`` DiT). The 8B DiT runs
out of DRAM on a single chip, so the heavy net runs **tensor-parallel across a
multi-chip mesh** (Megatron-1D over a ``(None, "model")`` mesh — the shard spec
the model-runner ``tensor_parallel-inference`` test validates). The precision-
sensitive text encoder, scheduler and VAE stay on CPU.

The reusable pipeline implementation is shared with the model-runner nightly
test and lives in ``tt_forge_models`` (``fibo/pytorch/pipeline.py``) — this
script is a thin runnable demo so the implementation isn't duplicated in
``examples/``, mirroring ``sd_v3_pipeline.py``.

FIBO is trained on structured JSON captions but also accepts plain text; here we
pass a JSON-style prompt as the model card's Generate example does. The gated
briaai/FIBO weights require accepting the bria-fibo license on Hugging Face and
authenticating via ``HF_TOKEN``.
"""

from pathlib import Path

import torch
import torch_xla.runtime as xr
from third_party.tt_forge_models.fibo.pytorch.pipeline import FiboConfig, FiboPipeline

# JSON-style structured prompt (FIBO's native caption format; see the model card).
PROMPT = (
    '{"subject":"a red panda astronaut floating above a glowing coral reef",'
    '"style_medium":"photograph","camera":"85mm prime, shallow depth of field",'
    '"lighting":"soft rim light with teal and warm highlights"}'
)
NEGATIVE_PROMPT = ""
# FIBO's model-card Generate example uses guidance_scale=5.0 / 50 steps. We run
# fewer denoise steps to fit the example budget while keeping the full native
# 1024x1024 geometry — flow-matching converges well at this step count.
NUM_INFERENCE_STEPS = 28
SEED = 42
OUTPUT_PATH = "fibo_output.png"


def save_image(image: torch.Tensor, filepath: str = OUTPUT_PATH):
    """Save a FIBO output image tensor ``(1, 3, H, W)`` in [0, 1] to ``filepath``."""
    from PIL import Image

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
    """Build the TP-sharded FIBO pipeline, generate one image, and save it."""
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
    print(f"Prompt: {PROMPT}")
    print(f"Generated image tensor: shape={tuple(image.shape)}, dtype={image.dtype}")
    print(
        f"Pixel range: [{image.min().item():.4f}, {image.max().item():.4f}], "
        f"mean={image.mean().item():.4f}"
    )
    print(f"Saved output image to {output_path}")


def main():
    image = run_fibo()
    post_process_output(image)


def test_fibo():
    """Generate one FIBO image on device and assert it is valid 1024x1024 RGB."""
    from PIL import Image

    xr.set_device_type("TT")

    output_path = "test_fibo_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    # A few denoise steps is enough to exercise the full TP-sharded device path;
    # the 8B DiT compile (not the step count) dominates wall time.
    image = run_fibo(output_path=output_path, num_inference_steps=4)

    assert torch.isfinite(image).all(), "FIBO output contains non-finite values"
    assert output_file.exists(), f"Output image {output_path} was not created"

    with Image.open(output_path) as img:
        width, height = img.size
    assert (width, height) == (1024, 1024), f"Expected 1024x1024, got {width}x{height}"

    print(f"FIBO output image created with resolution {width}x{height}")


if __name__ == "__main__":
    main()
