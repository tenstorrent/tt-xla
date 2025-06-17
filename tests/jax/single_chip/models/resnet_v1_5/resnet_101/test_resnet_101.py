# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode

from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    incorrect_result,
)

from ..tester import ResNetTester, ResNetVariant

MODEL_VARIANT = ResNetVariant.RESNET_101
MODEL_NAME = build_model_name(
    Framework.JAX,
    "resnet_v1.5",
    "101",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> ResNetTester:
    return ResNetTester(MODEL_VARIANT)


@pytest.fixture
def training_tester() -> ResNetTester:
    return ResNetTester(MODEL_VARIANT, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
@pytest.mark.xfail(
    reason=incorrect_result(
        "PCC comparison failed. Calculated: pcc=-0.08088425546884537. Required: pcc=0.99 "
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
def test_resnet_v1_5_101_inference(inference_tester: ResNetTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_resnet_v1_5_101_training(training_tester: ResNetTester):
    training_tester.test()
