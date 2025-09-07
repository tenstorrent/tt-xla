# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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
    failed_ttmlir_compilation,
)

from .tester import AlexNetTester
from third_party.tt_forge_models.alexnet.image_classification.jax import ModelVariant

VARIANT = ModelVariant.CUSTOM

MODEL_NAME = build_model_name(
    Framework.JAX,
    "alexnet",
    "custom",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.CUSTOM,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> AlexNetTester:
    return AlexNetTester(VARIANT)


@pytest.fixture
def training_tester() -> AlexNetTester:
    return AlexNetTester(VARIANT, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "failed to legalize operation 'ttir.gather' "
        "https://github.com/tenstorrent/tt-xla/issues/318"
    )
)
def test_alexnet_inference(inference_tester: AlexNetTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_alexnet_training(training_tester: AlexNetTester):
    training_tester.test()
