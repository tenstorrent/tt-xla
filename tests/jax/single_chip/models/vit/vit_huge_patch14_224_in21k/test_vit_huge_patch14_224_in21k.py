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

from ..tester import ViTTester

MODEL_PATH = "google/vit-huge-patch14-224-in21k"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "vit",
    "huge_patch14_224_in21k",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> ViTTester:
    return ViTTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> ViTTester:
    return ViTTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
def test_vit_huge_patch14_224_in21k_inference(
    inference_tester: ViTTester,
):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_vit_huge_patch14_224_in21k_training(training_tester: ViTTester):
    training_tester.test()
