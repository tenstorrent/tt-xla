# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_ttmlir_compilation,
)

from .tester import SWINTester
from third_party.tt_forge_models.swin.image_classification.pytorch import ModelVariant

VARIANT_NAME = ModelVariant.SWIN_T

MODEL_NAME = build_model_name(
    Framework.TORCH,
    "swin",
    "base",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.TORCH_HUB,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> SWINTester:
    return SWINTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> SWINTester:
    return SWINTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        " Error: torch_xla/csrc/aten_xla_bridge.cpp:110 : Check failed: xtensor "
        "https://github.com/tenstorrent/tt-xla/issues/795"
    )
)
def test_torch_swin_inference(inference_tester: SWINTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_swin_training(training_tester: SWINTester):
    training_tester.test()
