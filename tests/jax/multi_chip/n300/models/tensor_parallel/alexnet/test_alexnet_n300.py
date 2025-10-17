# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode, enable_shardy, DynamicJaxMultiChipModelTester
from utils import BringupStatus, Category

from third_party.tt_forge_models.alexnet.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)
from third_party.tt_forge_models.config import Parallelism

VARIANT_NAME = ModelVariant.CUSTOM_1X2
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> DynamicJaxMultiChipModelTester:
    model_loader = ModelLoader(VARIANT_NAME)
    return DynamicJaxMultiChipModelTester(
        model_loader=model_loader,
        run_mode=RunMode.INFERENCE,
        num_devices=2,
    )


@pytest.fixture
def training_tester() -> DynamicJaxMultiChipModelTester:
    model_loader = ModelLoader(VARIANT_NAME)
    return DynamicJaxMultiChipModelTester(
        model_loader=model_loader,
        run_mode=RunMode.TRAINING,
        num_devices=2,
    )


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
def test_alexnet_multichip_n300_inference(inference_tester: DynamicJaxMultiChipModelTester):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.TENSOR_PARALLEL,
)
def test_alexnet_multichip_n300_inference_shardy(
    inference_tester: DynamicJaxMultiChipModelTester,
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
def test_alexnet_multichip_n300_training(training_tester: DynamicJaxMultiChipModelTester):
    training_tester.test()
