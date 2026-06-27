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
    incorrect_result,
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


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
@pytest.mark.xfail(
    reason=incorrect_result(
        "Atol comparison failed. Calculated: atol=743930.6875. Required: atol=0.16 "
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
def test_deepseek_r1_qwen_inference(inference_tester: DeepseekTester):
    inference_tester.test()


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
