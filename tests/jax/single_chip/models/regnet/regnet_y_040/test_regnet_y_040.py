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
    failed_runtime,
)

from ..tester import RegNetTester

MODEL_PATH = "facebook/regnet-y-040"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "regnet",
    "y_040",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> RegNetTester:
    return RegNetTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> RegNetTester:
    return RegNetTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.xfail(
    reason=failed_runtime(
        "Out of Memory: Not enough space to allocate 802816 B L1 buffer "
        "across 1 banks, where each bank needs to store 802816 B "
        "(https://github.com/tenstorrent/tt-xla/issues/187)"
    )
)
def test_regnet_y_040_inference(inference_tester: RegNetTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_regnet_y_040_training(training_tester: RegNetTester):
    training_tester.test()
