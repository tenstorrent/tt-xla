# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from infra.multichip_utils import enable_shardy

from tests.jax.multi_chip.n300.models.tensor_parallel.alexnet.tester import (
    AlexNetMultichipTester,
)
from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_runtime,
)

MODEL_NAME = build_model_name(
    Framework.JAX,
    "alexnet",
    "multichip_llmbox_1x4",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.CUSTOM,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> AlexNetMultichipTester:
    return AlexNetMultichipTester(run_mode=RunMode.INFERENCE, num_devices=4)


@pytest.fixture
def training_tester() -> AlexNetMultichipTester:
    return AlexNetMultichipTester(run_mode=RunMode.TRAINING, num_devices=4)


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
def test_alexnet_multichip_llmbox_1x4_inference(
    inference_tester: AlexNetMultichipTester,
):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
)
def test_alexnet_multichip_llmbox_1x4_inference_shardy(
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
def test_alexnet_multichip_llmbox_1x4_training(training_tester: AlexNetMultichipTester):
    training_tester.test()
