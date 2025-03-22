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

from ..tester import Mistral7BV02Tester

MODEL_PATH = "unsloth/mistral-7b-instruct-v0.3"
MODEL_GROUP = ModelGroup.RED
MODEL_NAME = build_model_name(
    Framework.JAX,
    "mistral-7b",
    "v0.3_instruct",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> Mistral7BV02Tester:
    return Mistral7BV02Tester(MODEL_PATH)


def training_tester() -> Mistral7BV02Tester:
    return Mistral7BV02Tester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=MODEL_GROUP,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_FE_COMPILATION,
)
@pytest.mark.skip(
    reason=failed_fe_compilation(
        "OOMs in CI (https://github.com/tenstorrent/tt-xla/issues/186)"
    )
)
def test_mistral_7b_v0_3_instruct_inference(inference_tester: Mistral7BV02Tester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=MODEL_GROUP,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_mistral_7b_v0_3_instruct_training(training_tester: Mistral7BV02Tester):
    training_tester.test()
