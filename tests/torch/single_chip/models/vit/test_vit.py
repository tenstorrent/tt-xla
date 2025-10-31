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

from third_party.tt_forge_models.vit.pytorch import ModelVariant

from .tester import VITTester

VARIANT_NAME = ModelVariant.LARGE

MODEL_NAME = build_model_name(
    Framework.TORCH,
    "vit",
    "large-patch16-224",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.TORCH_HUB,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> VITTester:
    return VITTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> VITTester:
    return VITTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_torch_vit_inference(inference_tester: VITTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_vit_training(training_tester: VITTester):
    training_tester.test()
