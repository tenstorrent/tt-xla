# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode

from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_fe_compilation,
)

from ..tester import Dinov2Tester

MODEL_PATH = "facebook/dinov2-giant"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "dinov2",
    "giant",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> Dinov2Tester:
    return Dinov2Tester(MODEL_PATH)


@pytest.fixture
def training_tester() -> Dinov2Tester:
    return Dinov2Tester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_FE_COMPILATION,
)
@pytest.mark.skip(
    reason=failed_fe_compilation(
        'AttributeError: "FlaxDinov2SwiGLUFFN" object has no attribute "hidden_features" '
        "(https://github.com/tenstorrent/tt-xla/issues/567)"
    )
)
def test_dinov2_giant_inference(inference_tester: Dinov2Tester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_dinov2_giant_training(training_tester: Dinov2Tester):
    training_tester.test()
