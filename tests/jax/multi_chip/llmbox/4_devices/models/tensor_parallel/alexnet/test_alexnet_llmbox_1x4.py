# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode, enable_shardy
from utils import (
    BringupStatus,
    Category,
)
from third_party.tt_forge_models.config import Parallelism

from tests.jax.multi_chip.n300.models.tensor_parallel.alexnet.tester import (
    AlexNetMultichipTester,
)
from third_party.tt_forge_models.alexnet.image_classification.jax import (
    ModelVariant,
    ModelLoader,
)

VARIANT_NAME = ModelVariant.CUSTOM_1X4
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)


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
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.TENSOR_PARALLEL,
    bringup_status=BringupStatus.PASSED,
)
def test_alexnet_multichip_llmbox_1x4_inference(
    inference_tester: AlexNetMultichipTester,
):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.TENSOR_PARALLEL,
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
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.TENSOR_PARALLEL,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_alexnet_multichip_llmbox_1x4_training(training_tester: AlexNetMultichipTester):
    training_tester.test()
