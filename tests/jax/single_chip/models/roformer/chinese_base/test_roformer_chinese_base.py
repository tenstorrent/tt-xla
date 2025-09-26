# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from tests.infra.comparators.comparison_config import ComparisonConfig, PccConfig
from utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
)

from ..tester import RoFormerTester

MODEL_PATH = "junnyu/roformer_chinese_base"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "roformer",
    "chinese_base",
    ModelTask.NLP_MASKED_LM,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> RoFormerTester:
    # Reduced PCC threshold - #1454
    return RoFormerTester(
        MODEL_PATH, comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.98))
    )


@pytest.fixture
def training_tester() -> RoFormerTester:
    return RoFormerTester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
def test_roformer_chinese_base_inference(inference_tester: RoFormerTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_roformer_chinese_base_training(training_tester: RoFormerTester):
    training_tester.test()
