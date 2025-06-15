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

from .tester import FlaxBertSentenceEmbeddingTester

MODEL_PATH = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "bert",
    "base-turkish-stsb",
    ModelTask.NLP_TRANSLATION,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxBertSentenceEmbeddingTester:
    return FlaxBertSentenceEmbeddingTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> FlaxBertSentenceEmbeddingTester:
    return FlaxBertSentenceEmbeddingTester(MODEL_PATH, RunMode.TRAINING)


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
        "PCC comparison failed. Calculated: pcc=-0.11274315416812897. Required: pcc=0.99"
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
def test_flax_bert_base_turkish_inference(inference_tester: FlaxBertSentenceEmbeddingTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_bert_base_turkish_training(training_tester: FlaxBertSentenceEmbeddingTester):
    training_tester.test()