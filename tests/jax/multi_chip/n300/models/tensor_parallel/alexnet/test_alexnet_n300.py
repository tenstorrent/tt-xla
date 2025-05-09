# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from infra.multichip_utils import enable_shardy

from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_runtime,
    incorrect_result,
)

from .tester import AlexNetMultichipTester

MODEL_NAME = build_model_name(
    Framework.JAX,
    "alexnet",
    "multichip_n300",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.CUSTOM,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> AlexNetMultichipTester:
    return AlexNetMultichipTester(run_mode=RunMode.INFERENCE)


@pytest.fixture
def training_tester() -> AlexNetMultichipTester:
    return AlexNetMultichipTester(run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.xfail(
    reason=incorrect_result(
        "Atol comparison failed. Calculated: atol=0.4999960660934448. Required: atol=0.16. "
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
def test_alexnet_multichip_n300_inference(inference_tester: AlexNetMultichipTester):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
)
@pytest.mark.xfail(
    reason=failed_runtime(
        "Assertion `meshShape.size() == 2' failed "
        "(https://github.com/tenstorrent/tt-mlir/issues/3130)"
    )
)
def test_alexnet_multichip_n300_inference_shardy(
    inference_tester: AlexNetMultichipTester,
):
    with enable_shardy(True):
        inference_tester.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_alexnet_multichip_n300_training(training_tester: AlexNetMultichipTester):
    training_tester.test()
