# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from pytest import MonkeyPatch
from utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
)

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.nsfw_image_detection.pytorch import ModelVariant

from .tester import NsfwImageDetectionTester

VARIANT_NAME = ModelVariant.BASE

MODEL_NAME = build_model_name(
    Framework.TORCH,
    "NSFWImageDetection",
    "Base",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def trace_tester(monkeypatch: MonkeyPatch) -> NsfwImageDetectionTester:
    monkeypatch.setenv("TT_RUNTIME_TRACE_REGION_SIZE", "10000000")

    compiler_config = CompilerConfig(optimization_level=1, enable_trace=True)
    return NsfwImageDetectionTester(VARIANT_NAME, compiler_config=compiler_config)


# ----- Tests -----


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_torch_nsfw_image_detection_inference(
    trace_tester: NsfwImageDetectionTester,
):
    trace_tester.test()
