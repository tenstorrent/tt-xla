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
    failed_fe_compilation,
)

from ..tester import WhisperTester

MODEL_PATH = "openai/whisper-medium"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "whisper",
    "medium",
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
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.skip(
    reason=failed_fe_compilation(
        "Segfault (https://github.com/tenstorrent/tt-xla/issues/546)"
    )
)
def test_whisper_medium_inference(inference_tester: WhisperTester):
    inference_tester.test()


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_whisper_medium_training(training_tester: WhisperTester):
    training_tester.test()
