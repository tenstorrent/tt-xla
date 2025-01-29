# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from infra import RunMode

from ..tester import MNISTCNNTester
from .model_implementation import MNISTCNNNoDropoutModel

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MNISTCNNTester:
    return MNISTCNNTester(MNISTCNNNoDropoutModel)


@pytest.fixture
def training_tester() -> MNISTCNNTester:
    return MNISTCNNTester(MNISTCNNNoDropoutModel, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.skip(
    reason='void mlir::OperationConverter::finalize(mlir::ConversionPatternRewriter &): Assertion `newValue && "replacement value not found"\' failed.'
)  # This is a segfault, marking it as xfail would bring down the whole test suite
def test_mnist_cnn_nodropout_inference(
    inference_tester: MNISTCNNTester,
):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_mnist_cnn_nodropout_training(
    training_tester: MNISTCNNTester,
):
    training_tester.test()
