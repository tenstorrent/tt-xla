# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Stable Diffusion 1.5 — nightly e2e pipeline test.

Runs the full text-to-image pipeline end-to-end: the UNet (the heavy net) runs
on Tenstorrent via ``torch.compile(backend="tt")`` while the precision-sensitive
CLIP text encoder, the scheduler and the VAE stay on CPU. The reusable pipeline
implementation lives in ``tt_forge_models`` and is shared with the image-gen
benchmark (``tests/benchmark/test_imagegen.py``).
"""

from pathlib import Path

import pytest
import torch_xla.runtime as xr
from infra import RunMode
from loguru import logger
from PIL import Image
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.stable_diffusion_1_5.pytorch.pipeline import (
    SD15Config,
    SD15Pipeline,
    save_image,
)

PROMPT = "a photo of a cat"
NEGATIVE_PROMPT = ""
SEED = 42
CFG_SCALE = 7.5
HEIGHT = 512
WIDTH = 512


def run_sd15_pipeline(
    output_path: str = "sd15_output.png",
    num_inference_steps: int = 50,
):
    """Run the Stable Diffusion 1.5 pipeline and save the output image."""
    # CLIP on CPU (precision-sensitive); UNet on TT.
    config = SD15Config(clip_on_tt=False)
    pipeline = SD15Pipeline(config=config)
    pipeline.setup()

    img = pipeline.generate(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        cfg_scale=CFG_SCALE,
        num_inference_steps=num_inference_steps,
        seed=SEED,
    )

    save_image(img, output_path)
    return output_path


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="StableDiffusion1_5_Pipeline",
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_stable_diffusion_1_5_pipeline():
    """Run the full Stable Diffusion 1.5 pipeline with the UNet on TT."""
    xr.set_device_type("TT")

    output_path = "sd15_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    run_sd15_pipeline(output_path=output_path, num_inference_steps=50)

    assert output_file.exists(), f"Output image {output_path} was not created"

    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"

    logger.info(f"Output image saved to {output_path} ({width}x{height})")
