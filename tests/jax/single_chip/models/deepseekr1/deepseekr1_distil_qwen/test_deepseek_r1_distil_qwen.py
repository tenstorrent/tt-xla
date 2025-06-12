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
    failed_fe_compilation
)

from ..tester import DeepseekTester

MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "deepseek",
    "r1-distill-qwen-1.5B",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> DeepseekTester:
    return DeepseekTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> DeepseekTester:
    return DeepseekTester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_FE_COMPILATION,
)

def test_deepseek_r1_qwen_inference(inference_tester: DeepseekTester):
    inference_tester.test()

@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_deepseek_r1_qwen_training(training_tester: DeepseekTester):
    training_tester.test()
