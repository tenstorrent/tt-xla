# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
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

from .tester import BinaryAddTester

VARIANT_NAME = "binary_add"


MODEL_NAME = build_model_name(
    Framework.TORCH,
    "binary_add",
    "base",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.TORCH_HUB,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> BinaryAddTester:
    return BinaryAddTester(VARIANT_NAME)


# @pytest.fixture
# def training_tester() -> BinaryAddTester:
#     return BinaryAddTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_torch_binary_add_inference(inference_tester: BinaryAddTester):
    inference_tester.test()


# @pytest.mark.nightly
# @pytest.mark.record_test_properties(
#     category=Category.MODEL_TEST,
#     model_name=MODEL_NAME,
#     model_group=ModelGroup.GENERALITY,
#     run_mode=RunMode.TRAINING,
# )
# @pytest.mark.skip(reason="Support for training not implemented")
# def test_torch_alextnet_training(training_tester: BinaryAddTester):
#     training_tester.test()
