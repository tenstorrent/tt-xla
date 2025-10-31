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
)

from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch import ModelVariant

from .tester import Qwen2_5Tester

VARIANT_NAME = ModelVariant.QWEN_2_5_1_5B

MODEL_NAME = build_model_name(
    Framework.TORCH,
    "qwen_2_5",
    "1.5B",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> Qwen2_5Tester:
    return Qwen2_5Tester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> Qwen2_5Tester:
    return Qwen2_5Tester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_torch_qwen_2_5_inference(inference_tester: Qwen2_5Tester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_qwen_2_5_training(training_tester: Qwen2_5Tester):
    training_tester.test()
