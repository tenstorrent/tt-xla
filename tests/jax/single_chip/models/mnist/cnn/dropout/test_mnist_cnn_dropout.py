# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from utils import (
    BringupStatus,
    Category,
    ExecutionPass,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_ttmlir_compilation,
)

from ..tester import MNISTCNNTester
from third_party.tt_forge_models.mnist.image_classification.jax import (
    ModelLoader,
    ModelArchitecture,
)

MODEL_NAME = build_model_name(
    Framework.JAX,
    "mnist",
    "cnn_dropout",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.CUSTOM,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MNISTCNNTester:
    return MNISTCNNTester(ModelArchitecture.CNN_DROPOUT)


@pytest.fixture
def training_tester() -> MNISTCNNTester:
    return MNISTCNNTester(ModelArchitecture.CNN_DROPOUT, run_mode=RunMode.TRAINING)


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
def test_mnist_cnn_dropout_inference(inference_tester: MNISTCNNTester):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
    execution_pass=ExecutionPass.FORWARD,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "error: failed to legalize operation 'stablehlo.bitcast_convert'"
        "https://github.com/tenstorrent/tt-mlir/issues/979"
    )
)
def test_mnist_cnn_dropout_training(training_tester: MNISTCNNTester):
    training_tester.test()
