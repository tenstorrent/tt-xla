# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable FIBO (briaai/FIBO) text-to-image example on Tenstorrent.

FIBO is BRIA AI's 8B-parameter DiT flow-matching text-to-image model (SmolLM3-3B
text encoder, Wan 2.2 VAE). The 8B DiT does not fit in a single chip's DRAM, so
the heavy net runs tensor-parallel (Megatron-1D over a ``(None, "model")`` mesh)
across the multi-chip mesh via ``torch.compile(backend="tt")``; the precision-
sensitive text encoder, scheduler and VAE stay on CPU.

The reusable pipeline lives in ``tt_forge_models`` (shared with the image-gen
benchmark, ``tests/benchmark/test_imagegen.py``) — this script is a thin runnable
demo so the implementation isn't duplicated in ``examples/``, mirroring
``sd_v3_pipeline.py``.
"""

import torch
import torch_xla
import torch_xla.runtime as xr
from PIL import Image

from third_party.tt_forge_models.fibo.pytorch.pipeline import FiboConfig, FiboPipeline

PROMPT = "An astronaut riding a green horse on a grassy plain, photorealistic, cinematic lighting, 8k"
NEGATIVE_PROMPT = None
# Native FIBO uses a 50-step flow-matching schedule; this example uses fewer
# denoise steps to fit the example stage's time budget. Resolution stays native
# 1024x1024 — only the step count is reduced (and noted in the report).
NUM_INFERENCE_STEPS = 8
SEED = 42
OUTPUT_PATH = "fibo_output.png"
# qb2 (Blackhole) aborts compilation at optimization_level >= 1 (OpModel grid
# mismatch), so the sharded DiT is compiled at optimization_level = 0. The
# image-gen harness defaults to opt=1, hence the explicit override here.
OPTIMIZATION_LEVEL = 0


def run_fibo(
    prompt: str = PROMPT,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    seed: int = SEED,
):
    """Build the FIBO pipeline on the mesh and generate one image.

    Returns:
        tuple: ``(image, pipeline)`` — ``image`` is a ``(1, 3, H, W)`` float
        tensor in ``[0, 1]``; ``pipeline`` carries per-step timings in
        ``pipeline._perf``.
    """
    torch_xla.set_custom_compile_options(
        {"optimization_level": OPTIMIZATION_LEVEL, "enable_trace": False}
    )

    pipeline = FiboPipeline(config=FiboConfig())
    pipeline.setup()

    image = pipeline.generate(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
    return image, pipeline


def save_image(image: torch.Tensor, filepath: str = OUTPUT_PATH) -> str:
    """Save a ``(1, 3, H, W)`` float tensor in ``[0, 1]`` as a PNG."""
    img = (torch.clamp(image, 0.0, 1.0) * 255.0).to(torch.uint8)
    img_np = img.detach().cpu().squeeze(0).numpy()  # (3, H, W)
    if img_np.shape[0] == 3:
        img_np = img_np.transpose(1, 2, 0)  # (H, W, C)
    Image.fromarray(img_np).save(filepath)
    return filepath


def post_process_output(image: torch.Tensor, pipeline, filepath: str = OUTPUT_PATH) -> str:
    """Save the generated image and print a human-readable summary."""
    path = save_image(image, filepath)
    _, _, height, width = image.shape
    print(f"FIBO generated a {width}x{height} image; saved to {path}")

    perf = getattr(pipeline, "_perf", None)
    if perf and perf.get("steps"):
        steps = perf["steps"]
        total = perf.get("total") or 0.0
        print(
            f"Transformer steps: {len(steps)} | "
            f"mean {sum(steps) / len(steps):.3f}s/step | "
            f"generate() wall {total:.1f}s"
        )
    return path


def test_fibo():
    """Smoke test: FIBO produces a finite, correctly-shaped 1024x1024 image."""
    xr.set_device_type("TT")

    image, _ = run_fibo(num_inference_steps=1)

    assert image.dim() == 4 and tuple(image.shape[:2]) == (
        1,
        3,
    ), f"unexpected shape {tuple(image.shape)}"
    assert tuple(image.shape[-2:]) == (
        1024,
        1024,
    ), f"expected 1024x1024, got {tuple(image.shape[-2:])}"
    assert torch.isfinite(image).all(), "image contains non-finite values"
    assert (
        image.min() >= 0.0 and image.max() <= 1.0 + 1e-3
    ), f"image out of [0, 1] range: [{image.min():.3f}, {image.max():.3f}]"
    print(
        f"test_fibo OK: image {tuple(image.shape)}, "
        f"range [{image.min():.3f}, {image.max():.3f}]"
    )


if __name__ == "__main__":
    xr.set_device_type("TT")
    generated_image, fibo_pipeline = run_fibo()
    post_process_output(generated_image, fibo_pipeline)
