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
    failed_runtime,
)

from ..tester import RegNetTester
from third_party.tt_forge_models.regnet.image_classification.jax import ModelVariant

VARIANT_NAME = ModelVariant.REGNET_Y_320
MODEL_NAME = build_model_name(
    Framework.JAX,
    "regnet",
    "y_320",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> RegNetTester:
    return RegNetTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> RegNetTester:
    return RegNetTester(VARIANT_NAME, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.large
@pytest.mark.xfail(
    reason=failed_runtime(
        "Out of Memory: Not enough space to allocate 56438554624 B L1 buffer "
        "across 1 banks, where each bank needs to store 4703215616 B "
        "(https://github.com/tenstorrent/tt-xla/issues/187)"
    )
)
def test_regnet_y_320_inference(inference_tester: RegNetTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_regnet_y_320_training(training_tester: RegNetTester):
    training_tester.test()
