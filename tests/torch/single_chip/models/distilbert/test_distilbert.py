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
    failed_ttmlir_compilation,
)

from third_party.tt_forge_models.distilbert.masked_lm.pytorch.loader import ModelVariant

from .tester import DistilBertTester

VARIANT_NAME = ModelVariant.DISTILBERT_BASE_CASED


MODEL_NAME = build_model_name(
    Framework.TORCH,
    "distilbert",
    "base-cased",
    ModelTask.NLP_TEXT_CLS,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> DistilBertTester:
    return DistilBertTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> DistilBertTester:
    return DistilBertTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_torch_distilbert_inference(inference_tester: DistilBertTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_distilbert_training(training_tester: DistilBertTester):
    training_tester.test()
