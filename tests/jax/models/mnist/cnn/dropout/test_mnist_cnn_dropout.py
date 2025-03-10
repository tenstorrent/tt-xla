# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
from infra import RunMode
from utils import record_model_test_properties

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


@pytest.mark.push
@pytest.mark.model_test
def test_mnist_cnn_dropout_inference(
    inference_tester: MNISTCNNTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, "mnist-cnn-dropout")

    inference_tester.test()


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.skip(reason="Support for training not implemented")
def test_mnist_cnn_nodropout_training(
    training_tester: MNISTCNNTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, "mnist-cnn-dropout")

    training_tester.test()
