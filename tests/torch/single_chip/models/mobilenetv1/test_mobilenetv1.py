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
)
from third_party.tt_forge_models.mobilenetv1.pytorch.loader import ModelVariant
from .tester import MobileNetV1Tester

VARIANT_NAME = ModelVariant.MOBILENET_V1_GITHUB


MODEL_NAME = build_model_name(
    Framework.TORCH,
    "mobilenet",
    "v1",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.GITHUB,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MobileNetV1Tester:
    return MobileNetV1Tester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> MobileNetV1Tester:
    return MobileNetV1Tester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_torch_mobilnetv1_inference(inference_tester: MobileNetV1Tester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_mobilnetv1_training(training_tester: MobileNetV1Tester):
    training_tester.test()
