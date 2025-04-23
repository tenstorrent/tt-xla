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

from ..tester import BlenderBotTester

MODEL_PATH = "facebook/blenderbot-400M-distill"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "blenderbot",
    "400m-distill",
    ModelTask.NLP_SUMMARIZATION,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> BlenderBotTester:
    return BlenderBotTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> BlenderBotTester:
    return BlenderBotTester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.skip(reason=failed_fe_compilation("Segfault"))
def test_blenderbot_400m_distill_inference(inference_tester: BlenderBotTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_blenderbot_400m_distill_training(training_tester: BlenderBotTester):
    training_tester.test()
