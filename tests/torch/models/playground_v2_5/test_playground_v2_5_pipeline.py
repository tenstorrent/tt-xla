# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Playground v2.5 — nightly e2e pipeline test.

Drives the runnable pipeline in ``examples/pytorch/playground_v2_5_pipeline.py``
and asserts the output image is produced at the expected dimensions.
"""

from pathlib import Path

import pytest
import torch_xla.runtime as xr
from infra import RunMode
from loguru import logger
from PIL import Image
from utils import BringupStatus, Category, ModelGroup

from examples.pytorch.playground_v2_5_pipeline import (
    HEIGHT,
    WIDTH,
    run_playground_v25_pipeline,
)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="PlaygroundV2_5_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_playground_v25_pipeline():
    """Run the full Playground v2.5 pipeline with all components on TT."""
    xr.set_device_type("TT")

    output_path = "playground_v2_5_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    run_playground_v25_pipeline(
        output_path=output_path,
        num_inference_steps=50,
    )

    assert output_file.exists(), f"Output image {output_path} was not created"

    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"

    logger.info(f"Output image saved to {output_path} ({width}x{height})")
