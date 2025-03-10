# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import runtime_fail

from ..tester import FlaxRobertaForMaskedLMTester

MODEL_PATH = "FacebookAI/roberta-base"
MODEL_NAME = "roberta-base"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxRobertaForMaskedLMTester:
    return FlaxRobertaForMaskedLMTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> FlaxRobertaForMaskedLMTester:
    return FlaxRobertaForMaskedLMTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.INFERENCE.value,
)
@pytest.mark.xfail(
    reason=runtime_fail(
        "Atol comparison failed. Calculated: atol=131044.359375. Required: atol=0.16"
    )
)
def test_flax_roberta_base_inference(inference_tester: FlaxRobertaForMaskedLMTester):
    inference_tester.test()


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.TRAINING.value,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_roberta_base_training(training_tester: FlaxRobertaForMaskedLMTester):
    training_tester.test()
