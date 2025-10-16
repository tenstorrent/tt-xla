# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category, ExecutionPass, failed_ttmlir_compilation

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.dinov2.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)

from ..tester import Dinov2Tester

VARIANT_NAME = ModelVariant.LARGE
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> Dinov2Tester:
    return Dinov2Tester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> Dinov2Tester:
    return Dinov2Tester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_dinov2_large_inference(inference_tester: Dinov2Tester):
    inference_tester.test()


@pytest.mark.training
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.TRAINING,
    execution_pass=ExecutionPass.BACKWARD,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "error: 'ttir.conv2d' op The output tensor height and width dimension (224, 224) do not match the expected dimensions (29, 29) "
        "https://github.com/tenstorrent/tt-mlir/issues/5304"
    )
)
def test_dinov2_large_training(training_tester: Dinov2Tester):
    training_tester.test()
