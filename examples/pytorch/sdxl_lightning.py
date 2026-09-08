# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SDXL-Lightning — text-to-image demo on Tenstorrent hardware.

Thin wrapper over the shared pipeline in ``tt_forge_models``, the same code the
benchmark (``tests/benchmark/test_imagegen.py::test_sdxl_lightning``) and the
PCC-gated e2e (``tests/torch/models/sdxl_lightning/``) run.

Run:
    python examples/pytorch/sdxl_lightning.py
"""

from pathlib import Path

import torch_xla.runtime as xr
from loguru import logger

from third_party.tt_forge_models.sdxl_lightning.pytorch.pipeline import (
    NUM_INFERENCE_STEPS,
    PROMPT,
    SEED,
    SDXLLightningConfig,
    SDXLLightningTTPipeline,
    save_image,
)


def run_sdxl_lightning(
    output_path: str = "sdxl_lightning_output.png",
    num_inference_steps: int = NUM_INFERENCE_STEPS,
):
    """Run the SDXL-Lightning pipeline end-to-end on TT and save the image."""
    pipeline = SDXLLightningTTPipeline(config=SDXLLightningConfig())
    pipeline.setup()

    img = pipeline.generate(
        prompt=PROMPT,
        num_inference_steps=num_inference_steps,
        seed=SEED,
    )

    save_image(img, output_path)
    return output_path


if __name__ == "__main__":
    xr.set_device_type("TT")
    output_path = "sdxl_lightning_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()
    run_sdxl_lightning(output_path=output_path)
    logger.info(f"Output image saved to {output_path}")
