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

from ..tester import GPT2Tester

MODEL_PATH = "openai-community/gpt2"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "gpt2",
    "base",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> GPT2Tester:
    return GPT2Tester(MODEL_PATH)


@pytest.fixture
def training_tester() -> GPT2Tester:
    return GPT2Tester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
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
        "Atol comparison failed. Calculated: atol=37.821876525878906. Required: atol=0.16"
    )
)
def test_gpt2_base_inference(inference_tester: GPT2Tester):
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
def test_gpt2_base_training(training_tester: GPT2Tester):
    training_tester.test()
