# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from gemma import gm

from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    incorrect_result,
)

from ..tester import GemmaTester

MODEL_PATH = gm.nn.Gemma3_1B()
MODEL_NAME = build_model_name(
    Framework.JAX,
    "gemma",
    "3_1b",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> GemmaTester:
    return GemmaTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> GemmaTester:
    return GemmaTester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
def test_gemma3_1b_inference(inference_tester: GemmaTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_gemma3_1b_training(training_tester: GemmaTester):
    training_tester.test()
