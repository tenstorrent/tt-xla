# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category

from third_party.tt_forge_models.bert.masked_lm.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)
from third_party.tt_forge_models.config import ModelInfo, Parallelism

from .tester import BertTester

VARIANT_NAME = ModelVariant.BERT_BASE_UNCASED

MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> BertTester:
    return BertTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> BertTester:
    return BertTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.PASSED,
)
def test_torch_bert_inference(inference_tester: BertTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_bert_training(training_tester: BertTester):
    training_tester.test()
