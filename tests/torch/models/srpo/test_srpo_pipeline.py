# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SRPO (tencent/SRPO) — nightly e2e text-to-image pipeline test.

SRPO is a ~12B FLUX.1-dev fine-tune whose transformer runs out of DRAM on a
single Wormhole chip, so the heavy net runs **tensor-parallel** across a
multi-chip mesh (Megatron-1D), while the FLUX.1-dev CLIP + T5 text encoders,
scheduler and taef1 VAE stay on CPU. This drives the shared ``SrpoPipeline``
from ``tt_forge_models`` (the same pipeline the image-gen benchmark uses)
end-to-end and asserts the saved image dimensions.
"""

from pathlib import Path

import pytest
import torch
from infra import RunMode
from PIL import Image
from utils import BringupStatus, Category

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.srpo.pytorch import ModelLoader, ModelVariant

# The SRPO e2e pipeline lives in tt-forge-models; skip cleanly until the
# submodule uplift brings in srpo/pytorch/pipeline.py (this PR carries no bump).
_pipeline = pytest.importorskip(
    "third_party.tt_forge_models.srpo.pytorch.pipeline",
    reason="requires tt-forge-models srpo/pytorch/pipeline.py (submodule uplift)",
)
SrpoConfig = _pipeline.SrpoConfig
SrpoPipeline = _pipeline.SrpoPipeline

VARIANT_NAME = ModelVariant.BASE
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

PROMPT = "An astronaut riding a horse in a futuristic city, highly detailed"
NUM_INFERENCE_STEPS = 28
SEED = 42
HEIGHT = 1024
WIDTH = 1024


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.TENSOR_PARALLEL,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_srpo_pipeline():
    """Run the SRPO pipeline (DiT tensor-parallel) and assert the output image."""
    pipeline = SrpoPipeline(config=SrpoConfig())
    pipeline.setup()

    image = pipeline.generate(
        prompt=PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    output_path = "test_srpo_pipeline_output.png"
    array = (image[0].float().clamp(0, 1) * 255).round().to(torch.uint8)
    array = array.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(output_path)

    assert Path(output_path).exists(), f"Output image {output_path} was not created"
    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"
