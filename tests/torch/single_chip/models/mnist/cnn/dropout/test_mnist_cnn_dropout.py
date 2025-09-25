# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from infra import Framework, RunMode
from tests.infra.testers.compiler_config import CompilerConfig
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
    Framework.TORCH,
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
def inference_tester_optimizer() -> MNISTCNNTester:
    return MNISTCNNTester(
        MNISTCNNDropoutModel,
        run_mode=RunMode.INFERENCE,
        compiler_config=CompilerConfig(
            enable_optimizer=True,
            enable_sharding=True,
            enable_fusing_conv2d_with_multiply_pattern=True,
        ),
    )


@pytest.fixture
def training_tester() -> MNISTCNNTester:
    return MNISTCNNTester(MNISTCNNDropoutModel, run_mode=RunMode.TRAINING)


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
def test_torch_mnist_cnn_dropout_inference(inference_tester: MNISTCNNTester):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_torch_mnist_cnn_dropout_inference_optimizer(
    inference_tester_optimizer: MNISTCNNTester,
):
    inference_tester_optimizer.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_mnist_cnn_dropout_training(training_tester: MNISTCNNTester):
    training_tester.test()
