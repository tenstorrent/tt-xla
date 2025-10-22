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

from third_party.tt_forge_models.mnist.image_classification.jax import ModelVariant

from ..tester import MNISTCNNTester

MODEL_NAME = build_model_name(
    Framework.JAX,
    "mnist",
    "cnn_nodropout",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.CUSTOM,
)


# ----- Fixtures -----


@pytest.fixture
def training_tester() -> MNISTCNNTester:
    return MNISTCNNTester(ModelVariant.CNN_BATCHNORM, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.training
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
        "error: failed to legalize operation 'stablehlo.select_and_scatter' "
        "https://github.com/tenstorrent/tt-mlir/issues/4687"
    )
)
def test_mnist_cnn_nodropout_training(training_tester: MNISTCNNTester):
    training_tester.test()
