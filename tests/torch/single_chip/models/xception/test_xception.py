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

from .tester import XceptionTester

VARIANT_NAME = "xception65"

MODEL_NAME = build_model_name(
    Framework.TORCH,
    "xception",
    "65",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.TORCH_HUB,
)


@pytest.fixture
def inference_tester() -> XceptionTester:
    return XceptionTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> XceptionTester:
    return XceptionTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_torch_xception_inference(inference_tester: XceptionTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_xception_training(training_tester: XceptionTester):
    training_tester.test()
