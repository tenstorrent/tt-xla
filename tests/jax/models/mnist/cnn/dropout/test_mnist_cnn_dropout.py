# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode

from ..tester import MNISTCNNTester
from .model_implementation import MNISTCNNDropoutModel

# ----- Fixtures -----

MODEL_NAME = "mnist-cnn-dropout"


@pytest.fixture
def inference_tester() -> MNISTCNNTester:
    return MNISTCNNTester(MNISTCNNDropoutModel)


@pytest.fixture
def training_tester() -> MNISTCNNTester:
    return MNISTCNNTester(MNISTCNNDropoutModel, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.INFERENCE.value,
)
def test_mnist_cnn_dropout_inference(inference_tester: MNISTCNNTester):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.TRAINING.value,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_mnist_cnn_nodropout_training(training_tester: MNISTCNNTester):
    training_tester.test()
