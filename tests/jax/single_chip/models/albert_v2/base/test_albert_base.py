# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
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

from ..tester import AlbertV2Tester

MODEL_PATH = "albert/albert-base-v2"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "albert_v2",
    "base",
    ModelTask.NLP_MASKED_LM,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> AlbertV2Tester:
    return AlbertV2Tester(MODEL_PATH)


@pytest.fixture
def training_tester() -> AlbertV2Tester:
    return AlbertV2Tester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_flax_albert_v2_base_inference(inference_tester: AlbertV2Tester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_albert_v2_base_training(training_tester: AlbertV2Tester):
    training_tester.test()
