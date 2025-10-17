# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode, enable_shardy, JaxMultichipModelTester
from utils import BringupStatus, Category

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.mnist.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)

VARIANT_NAME = ModelVariant.MLP_CUSTOM_1X2
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)


# ----- Fixtures -----


@pytest.fixture
def inference_tester(request) -> JaxMultichipModelTester:
    model_loader = ModelLoader(VARIANT_NAME, hidden_sizes=request.param)
    return JaxMultichipModelTester(
        model_loader=model_loader,
        run_mode=RunMode.INFERENCE,
        num_devices=2,
    )


@pytest.fixture
def training_tester(request) -> JaxMultichipModelTester:
    model_loader = ModelLoader(VARIANT_NAME, hidden_sizes=request.param)
    return JaxMultichipModelTester(
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
@pytest.mark.parametrize(
    "inference_tester", [(1024, 512, 256)], indirect=True, ids=lambda val: f"{val}"
)
def test_mnist_mlp_multichip_n300_inference(inference_tester: JaxMultichipModelTester):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.TENSOR_PARALLEL,
)
@pytest.mark.parametrize(
    "inference_tester", [(1024, 512, 256)], indirect=True, ids=lambda val: f"{val}"
)
def test_mnist_mlp_multichip_n300_inference_shardy(
    inference_tester: JaxMultichipModelTester,
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
@pytest.mark.parametrize(
    "training_tester", [(1024, 512, 256)], indirect=True, ids=lambda val: f"{val}"
)
def test_mnist_mlp_multichip_n300_training(training_tester: JaxMultichipModelTester):
    training_tester.test()
