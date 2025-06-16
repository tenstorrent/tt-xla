# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from infra.multichip_utils import enable_shardy

from tests.jax.multi_chip.n300.models.tensor_parallel.mnist_mlp.tester import (
    MnistMLPMultichipTester,
)
from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
)

MODEL_NAME = build_model_name(
    Framework.JAX,
    "mnist",
    "mlp_multichip_llmbox_1x4",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.CUSTOM,
)


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
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
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
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
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
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_mnist_mlp_multichip_llmbox_1x4_training(
    training_tester: MnistMLPMultichipTester,
):
    training_tester.test()
