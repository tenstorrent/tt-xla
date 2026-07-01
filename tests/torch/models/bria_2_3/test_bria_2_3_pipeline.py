# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""BRIA 2.3 — nightly e2e pipeline test.

Runs the full text-to-image pipeline end-to-end: the SDXL-class UNet (the heavy
net) runs on Tenstorrent via ``torch.compile(backend="tt")`` while the
precision-sensitive CLIP text encoders, the scheduler and the VAE stay on CPU.
The reusable pipeline implementation lives in ``tt_forge_models`` and is shared
with the image-gen benchmark (``tests/benchmark/test_imagegen.py``).
"""

from pathlib import Path

import pytest
import torch_xla.runtime as xr
from infra import RunMode
from loguru import logger
from PIL import Image
from utils import BringupStatus, Category, ModelGroup

# The BRIA 2.3 pipeline implementation lands via a tt_forge_models submodule
# uplift (tt-forge-models#720), which is intentionally not bumped in this PR.
# Until that uplift, importorskip skips this module at collection time so it
# doesn't break `pytest --collect-only` on tests/torch; the test runs once the
# pipeline is present.
_bria_pipeline = pytest.importorskip(
    "third_party.tt_forge_models.bria_2_3.pytorch.pipeline",
    reason="BRIA 2.3 pipeline pending tt_forge_models uplift (tt-forge-models#720)",
)
Bria23Config = _bria_pipeline.Bria23Config
Bria23Pipeline = _bria_pipeline.Bria23Pipeline
save_image = _bria_pipeline.save_image

PROMPT = (
    "A portrait of a Beautiful and playful ethereal singer, "
    "golden designs, highly detailed, blurry background"
)
NEGATIVE_PROMPT = ""
SEED = 42
GUIDANCE_SCALE = 5.0
HEIGHT = 1024
WIDTH = 1024


def run_bria_2_3_pipeline(
    output_path: str = "bria_2_3_output.png",
    num_inference_steps: int = 50,
):
    """Run the BRIA 2.3 pipeline and save the output image."""
    # Text encoders, scheduler and VAE on CPU (precision-sensitive); UNet on TT.
    config = Bria23Config()
    pipeline = Bria23Pipeline(config=config)
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
    model_name="Bria2_3_Pipeline",
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_bria_2_3_pipeline():
    """Run the full BRIA 2.3 pipeline with the UNet on TT."""
    xr.set_device_type("TT")

    output_path = "bria_2_3_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    run_bria_2_3_pipeline(output_path=output_path, num_inference_steps=50)

    assert output_file.exists(), f"Output image {output_path} was not created"

    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"

    logger.info(f"Output image saved to {output_path} ({width}x{height})")
