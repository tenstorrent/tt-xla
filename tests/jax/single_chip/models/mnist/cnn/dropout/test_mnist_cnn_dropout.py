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

from ..tester import MNISTCNNTester
from .model_implementation import MNISTCNNDropoutModel

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
    return MNISTCNNTester(MNISTCNNDropoutModel)


@pytest.fixture
def training_tester() -> MNISTCNNTester:
    return MNISTCNNTester(MNISTCNNDropoutModel, run_mode=RunMode.TRAINING)


# ----- Tests -----


# @pytest.mark.push
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


# @pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_mnist_cnn_dropout_training(training_tester: MNISTCNNTester):
    training_tester.test()
