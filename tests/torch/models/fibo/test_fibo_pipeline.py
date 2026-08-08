# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FIBO (briaai/FIBO) — nightly e2e text-to-image pipeline test.

FIBO is an 8B-parameter DiT text-to-image model whose transformer runs out of
DRAM on a single Wormhole chip, so the heavy net runs **tensor-parallel** across
a multi-chip mesh (Megatron-1D), while the SmolLM3 text encoder, scheduler and
Wan 2.2 VAE stay on CPU. This drives the shared ``FiboPipeline`` from
``tt_forge_models`` (the same pipeline the image-gen benchmark uses) end-to-end
and asserts the saved image dimensions.
"""

from pathlib import Path

import pytest
import torch
from infra import RunMode
from PIL import Image
from utils import BringupStatus, Category

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.fibo.pytorch import ModelLoader, ModelVariant

# The FIBO e2e pipeline lives in tt-forge-models; skip cleanly until the
# submodule uplift brings in fibo/pytorch/pipeline.py (this PR carries no bump).
_pipeline = pytest.importorskip(
    "third_party.tt_forge_models.fibo.pytorch.pipeline",
    reason="requires tt-forge-models fibo/pytorch/pipeline.py (submodule uplift)",
)
FiboConfig = _pipeline.FiboConfig
FiboPipeline = _pipeline.FiboPipeline

VARIANT_NAME = ModelVariant.BASE
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

PROMPT = (
    '{"subject":"a hyper-detailed, ultra-fluffy owl perched in moonlit trees",'
    '"style_medium":"photograph","camera":"85mm prime, shallow depth of field",'
    '"lighting":"cool moonlight with subtle silver highlights"}'
)
NUM_INFERENCE_STEPS = 50
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
def test_fibo_pipeline():
    """Run the FIBO pipeline (DiT tensor-parallel) and assert the output image."""
    pipeline = FiboPipeline(config=FiboConfig())
    pipeline.setup()

    image = pipeline.generate(
        prompt=PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    output_path = "test_fibo_pipeline_output.png"
    array = (image[0].float().clamp(0, 1) * 255).round().to(torch.uint8)
    array = array.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(output_path)

    assert Path(output_path).exists(), f"Output image {output_path} was not created"
    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"
