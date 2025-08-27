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

from third_party.tt_forge_models.unet.pytorch import ModelVariant
from .tester import UNETTester

VARIANT_NAME = ModelVariant.CARVANA_UNET

MODEL_NAME = build_model_name(
    Framework.TORCH,
    "unet",
    "base",
    ModelTask.CV_IMAGE_SEG,
    ModelSource.CUSTOM,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> UNETTester:
    return UNETTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> UNETTester:
    return UNETTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_torch_unet_inference(inference_tester: UNETTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_unet_training(training_tester: UNETTester):
    training_tester.test()
