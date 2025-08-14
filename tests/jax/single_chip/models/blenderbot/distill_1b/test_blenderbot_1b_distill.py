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

from ..tester import BlenderBotTester

MODEL_PATH = "facebook/blenderbot-1B-distill"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "blenderbot",
    "1b-distill",
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
@pytest.mark.large
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "Failed to legalize operation 'ttir.scatter' "
        "https://github.com/tenstorrent/tt-xla/issues/911"
    )
)
def test_blenderbot_1b_distill_inference(inference_tester: BlenderBotTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.large
@pytest.mark.skip(reason="Support for training not implemented")
def test_blenderbot_1b_distill_training(training_tester: BlenderBotTester):
    training_tester.test()
