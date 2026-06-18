# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Stable Diffusion 3 Medium — nightly e2e pipeline test.

Runs the full text-to-image pipeline end-to-end: the MMDiT transformer (the
heavy net) runs on Tenstorrent via ``torch.compile(backend="tt")`` while the
three text encoders (two CLIP + T5), the scheduler and the VAE stay on CPU. The
reusable pipeline implementation lives in ``tt_forge_models`` and is shared with
the image-gen benchmark (``tests/benchmark/test_imagegen.py``).
"""

from pathlib import Path

import pytest
import torch_xla.runtime as xr
from infra import RunMode
from loguru import logger
from PIL import Image
from utils import BringupStatus, Category, ModelGroup

# The SD3 pipeline implementation lands via a tt_forge_models submodule uplift
# (tt-forge-models#720), which is intentionally not bumped in this PR. Until that
# uplift, importorskip skips this module at collection time so it doesn't break
# `pytest --collect-only` on tests/torch; the test runs once the pipeline is present.
_sd3_pipeline = pytest.importorskip(
    "third_party.tt_forge_models.stable_diffusion_3.pytorch.pipeline",
    reason="SD3 pipeline pending tt_forge_models uplift (tt-forge-models#720)",
)
SD3Config = _sd3_pipeline.SD3Config
SD3Pipeline = _sd3_pipeline.SD3Pipeline
save_image = _sd3_pipeline.save_image

PROMPT = "An astronaut riding a green horse"
NEGATIVE_PROMPT = ""
SEED = 42
GUIDANCE_SCALE = 7.0
HEIGHT = 1024
WIDTH = 1024


def run_sd3_pipeline(
    output_path: str = "sd3_output.png",
    num_inference_steps: int = 28,
):
    """Run the Stable Diffusion 3 Medium pipeline and save the output image."""
    # Transformer on TT; text encoders, scheduler and VAE on CPU.
    config = SD3Config()
    pipeline = SD3Pipeline(config=config)
    pipeline.setup()

    img = pipeline.generate(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        guidance_scale=GUIDANCE_SCALE,
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
    model_name="StableDiffusion3_Medium_Pipeline",
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_stable_diffusion_3_pipeline():
    """Run the full Stable Diffusion 3 Medium pipeline with the transformer on TT."""
    xr.set_device_type("TT")

    output_path = "sd3_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    run_sd3_pipeline(output_path=output_path, num_inference_steps=28)

    assert output_file.exists(), f"Output image {output_path} was not created"

    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"

    logger.info(f"Output image saved to {output_path} ({width}x{height})")
