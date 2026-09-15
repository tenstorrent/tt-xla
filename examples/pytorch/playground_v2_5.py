# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Playground v2.5 — text-to-image demo on Tenstorrent hardware.

Thin wrapper over the shared pipeline in ``tt_forge_models``, the same code the
benchmark (``tests/benchmark/test_imagegen.py::test_playground_v2_5``) and the
PCC-gated e2e (``tests/torch/models/playground_v2_5/``) run.

Run:
    python examples/pytorch/playground_v2_5.py
"""

from pathlib import Path

import torch_xla.runtime as xr
from loguru import logger

from third_party.tt_forge_models.playground_v2_5.pytorch.pipeline import (
    GUIDANCE_SCALE,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    PROMPT,
    SEED,
    PlaygroundV25Config,
    PlaygroundV25TTPipeline,
    save_image,
)


def run_playground_v25(
    output_path: str = "playground_v2_5_output.png",
    num_inference_steps: int = NUM_INFERENCE_STEPS,
):
    """Run the Playground v2.5 pipeline end-to-end on TT and save the image."""
    pipeline = PlaygroundV25TTPipeline(config=PlaygroundV25Config())
    pipeline.setup()

    img = pipeline.generate(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        cfg_scale=GUIDANCE_SCALE,
        num_inference_steps=num_inference_steps,
        seed=SEED,
    )

    save_image(img, output_path)
    return output_path


if __name__ == "__main__":
    xr.set_device_type("TT")
    output_path = "playground_v2_5_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()
    run_playground_v25(output_path=output_path)
    logger.info(f"Output image saved to {output_path}")
