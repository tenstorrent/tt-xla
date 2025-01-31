# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from infra import RunMode

from ..tester import MNISTCNNTester
from .model_implementation import MNISTCNNDropoutModel

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MNISTCNNTester:
    return MNISTCNNTester(MNISTCNNDropoutModel)


@pytest.fixture
def training_tester() -> MNISTCNNTester:
    return MNISTCNNTester(MNISTCNNDropoutModel, RunMode.TRAINING)


# ----- Tests -----


def test_mnist_cnn_dropout_inference(
    inference_tester: MNISTCNNTester,
):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_mnist_cnn_nodropout_training(
    training_tester: MNISTCNNTester,
):
    training_tester.test()
