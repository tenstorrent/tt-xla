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

from ..tester import RoFormerTester

MODEL_PATH = "junnyu/roformer_chinese_small"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "roformer",
    "chinese_small",
    ModelTask.NLP_MASKED_LM,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> RoFormerTester:
    return RoFormerTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> RoFormerTester:
    return RoFormerTester(MODEL_PATH, run_mode=RunMode.TRAINING)


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
        "Atol comparison failed. Calculated: atol=11.5. Required: atol=0.1 "
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
def test_roformer_chinese_small_inference(inference_tester: RoFormerTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_roformer_chinese_small_training(training_tester: RoFormerTester):
    training_tester.test()
