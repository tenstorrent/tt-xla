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
)

from third_party.tt_forge_models.bert.masked_lm.jax import ModelVariant

from ..tester import FlaxBertForMaskedLMTester

MODEL_VARIANT = ModelVariant.LARGE
MODEL_NAME = build_model_name(
    Framework.JAX,
    "bert",
    "large",
    ModelTask.NLP_MASKED_LM,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxBertForMaskedLMTester:
    return FlaxBertForMaskedLMTester(MODEL_VARIANT)


@pytest.fixture
def training_tester() -> FlaxBertForMaskedLMTester:
    return FlaxBertForMaskedLMTester(MODEL_VARIANT, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_flax_bert_large_inference(inference_tester: FlaxBertForMaskedLMTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_bert_large_training(training_tester: FlaxBertForMaskedLMTester):
    training_tester.test()
