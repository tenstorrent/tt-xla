# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Infinity 2B — nightly e2e text-to-image pipeline test.

Infinity is an autoregressive next-scale-prediction text-to-image model whose 2B
transformer runs **tensor-parallel** across a multi-chip mesh , while the T5-XL text encoder, multinomial sampling and
BSQ-VAE decode stay on CPU. This drives the shared ``InfinityPipeline`` from
``tt_forge_models`` (the same pipeline the image-gen benchmark uses) end-to-end
and asserts the saved image dimensions.
"""

from pathlib import Path

import pytest
import torch
import torch_xla.runtime as xr
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from loguru import logger
from PIL import Image
from utils import BringupStatus, Category

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.infinity.pytorch import ModelLoader, ModelVariant

# The Infinity e2e pipeline lives in tt-forge-models; skip cleanly until the
# submodule uplift brings in infinity/pytorch/src/pipeline.py.
_pipeline = pytest.importorskip(
    "third_party.tt_forge_models.infinity.pytorch.src.pipeline",
    reason="requires tt-forge-models infinity/pytorch/src/pipeline.py (submodule uplift)",
)
InfinityConfig = _pipeline.InfinityConfig
InfinityPipeline = _pipeline.InfinityPipeline

VARIANT_NAME = ModelVariant.INFINITY_2B
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

PROMPT = "A fantasy landscape with mountains and rivers"
SEED = 42
HEIGHT = _pipeline.HEIGHT
WIDTH = _pipeline.WIDTH
# Lowest observed per-scale PCC is ~0.988 (the large scale-12/13 forwards);
# 0.98 keeps a small margin below that while still gating real regressions.
PCC_THRESHOLD = 0.98


# The transformer is the only Infinity component on TT (T5 text encode, sampling
# and BSQ-VAE decode stay on CPU). After every per-scale, per-CFG-branch TT
# transformer forward the pipeline feeds the same inputs to a fp32 CPU twin and
# hands both logits to this hook, which checks PCC and fails fast below threshold.
_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


def _pcc_hook(tag: str, device_logits, golden_logits) -> None:
    pcc = _pcc(device_logits, golden_logits)
    logger.info(f"[PCC] {tag}: pcc={pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"{tag} PCC {pcc:.6f} below threshold {PCC_THRESHOLD}"


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.lb_blackhole
@pytest.mark.tensor_parallel
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.TENSOR_PARALLEL,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_infinity_pipeline():
    """Run the Infinity 2B pipeline (transformer 8-way TP) with per-scale PCC checks.

    Every per-scale TT transformer forward is checked inline against an fp32 CPU
    twin fed the same inputs (via ``_pcc_hook``); the run fails fast the moment
    any scale's PCC drops below ``PCC_THRESHOLD``. The pipeline still generates
    the image from the TT outputs, and the output dimensions are asserted below.
    """
    xr.set_device_type("TT")

    pipeline = InfinityPipeline(config=InfinityConfig())
    pipeline.setup()

    image = pipeline.generate(prompt=PROMPT, seed=SEED, pcc_hook=_pcc_hook)

    output_path = "infinity_pipeline_output.png"
    # BSQ-VAE decode returns RGB in [-1, 1]; map to [0, 1] before saving.
    array = (image / 2 + 0.5).clamp(0, 1)
    array = (array[0].float() * 255).round().to(torch.uint8)
    array = array.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(output_path)

    assert Path(output_path).exists(), f"Output image {output_path} was not created"
    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"
