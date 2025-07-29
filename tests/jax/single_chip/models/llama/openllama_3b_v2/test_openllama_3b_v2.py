# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
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

from ..tester import LLamaTester

MODEL_PATH = "openlm-research/open_llama_3b_v2"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "open_llama_v2",
    "3b",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> LLamaTester:
    return LLamaTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> LLamaTester:
    return LLamaTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
@pytest.mark.large
@pytest.mark.xfail(
    reason=incorrect_result(
        "AssertionError: PCC comparison failed. Calculated: pcc=0.9683969616889954. Required: pcc=0.99."
    )
)
def test_openllama3b_inference(inference_tester: LLamaTester):
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
def test_openllama3b_training(training_tester: LLamaTester):
    training_tester.test()
