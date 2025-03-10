# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import runtime_fail

from ..tester import FlaxBertForMaskedLMTester

MODEL_PATH = "google-bert/bert-base-uncased"
MODEL_NAME = "bert-base"

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxBertForMaskedLMTester:
    return FlaxBertForMaskedLMTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> FlaxBertForMaskedLMTester:
    return FlaxBertForMaskedLMTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.INFERENCE.value,
)
@pytest.mark.xfail(
    reason=(
        runtime_fail(
            "Atol comparison failed. Calculated: atol=131025.0078125. Required: atol=0.16."
        )
    )
)
def test_flax_bert_base_inference(inference_tester: FlaxBertForMaskedLMTester):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.TRAINING.value,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_bert_base_training(training_tester: FlaxBertForMaskedLMTester):
    training_tester.test()
