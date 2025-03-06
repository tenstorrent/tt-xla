# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
from infra import RunMode
from utils import accuracy_fail, record_model_test_properties

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


@pytest.mark.nightly
@pytest.mark.xfail(
    reason=accuracy_fail(
        "Atol comparison failed. Calculated: atol=131044.359375. Required: atol=0.16"
    )
)
def test_flax_roberta_base_inference(
    inference_tester: FlaxRobertaForMaskedLMTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_roberta_base_training(
    training_tester: FlaxRobertaForMaskedLMTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    training_tester.test()
