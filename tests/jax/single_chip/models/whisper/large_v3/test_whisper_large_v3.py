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
    incorrect_result,
)

from ..tester import WhisperTester

MODEL_PATH = "openai/whisper-large-v3"
MODEL_GROUP = ModelGroup.RED
MODEL_NAME = build_model_name(
    Framework.JAX,
    "whisper",
    "large_v3",
    ModelTask.AUDIO_CLS,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> WhisperTester:
    return WhisperTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> WhisperTester:
    return WhisperTester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=MODEL_GROUP,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.large
@pytest.mark.xfail(
    reason=incorrect_result(
        "PCC comparison failed. Calculated: pcc=-1.0000001192092896. Required: pcc=0.99. "
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
def test_whisper_large_v3_inference(inference_tester: WhisperTester):
    inference_tester.test()


@pytest.mark.training
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=MODEL_GROUP,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.large
@pytest.mark.skip(reason="Support for training not implemented")
def test_whisper_large_v3_training(training_tester: WhisperTester):
    training_tester.test()
