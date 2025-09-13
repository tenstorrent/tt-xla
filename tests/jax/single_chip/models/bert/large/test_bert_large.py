# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import (
    BringupStatus,
    Category,
)
from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.bert.masked_lm.jax import (
    ModelVariant,
    ModelLoader,
)
from ..tester import FlaxBertForMaskedLMTester

VARIANT_NAME = ModelVariant.LARGE

MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxBertForMaskedLMTester:
    return FlaxBertForMaskedLMTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> FlaxBertForMaskedLMTester:
    return FlaxBertForMaskedLMTester(VARIANT_NAME, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.PASSED,
)
def test_flax_bert_large_inference(inference_tester: FlaxBertForMaskedLMTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_bert_large_training(training_tester: FlaxBertForMaskedLMTester):
    training_tester.test()
