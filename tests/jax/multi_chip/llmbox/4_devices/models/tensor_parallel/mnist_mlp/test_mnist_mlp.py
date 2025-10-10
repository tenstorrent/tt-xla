# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode, enable_shardy
from utils import BringupStatus, Category

from tests.jax.multi_chip.n300.models.tensor_parallel.mnist_mlp.tester import (
    MnistMLPMultichipTester,
)
from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.mnist.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)

VARIANT_NAME = ModelVariant.MLP_CUSTOM_1X4
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)


# ----- Fixtures -----


@pytest.fixture
def inference_tester(request) -> MnistMLPMultichipTester:
    return MnistMLPMultichipTester(
        request.param, run_mode=RunMode.INFERENCE, num_devices=4
    )


@pytest.fixture
def training_tester(request) -> MnistMLPMultichipTester:
    return MnistMLPMultichipTester(
        request.param, run_mode=RunMode.TRAINING, num_devices=4
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
def test_mnist_mlp_multichip_llmbox_1x4_inference(
    inference_tester: MnistMLPMultichipTester,
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
@pytest.mark.parametrize(
    "inference_tester", [(1024, 512, 256)], indirect=True, ids=lambda val: f"{val}"
)
def test_mnist_mlp_multichip_llmbox_1x4_inference_shardy(
    inference_tester: MnistMLPMultichipTester,
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
def test_mnist_mlp_multichip_llmbox_1x4_training(
    training_tester: MnistMLPMultichipTester,
):
    training_tester.test()
