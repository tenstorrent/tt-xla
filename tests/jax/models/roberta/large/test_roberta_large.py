# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
from infra import RunMode
from utils import record_model_test_properties, runtime_fail

from ..tester import FlaxRobertaForMaskedLMTester

MODEL_PATH = "FacebookAI/roberta-large"
MODEL_NAME = "roberta-large"

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
    reason=runtime_fail(
        "Host data with total size 20B does not match expected size 10B of device buffer! "
        "(https://github.com/tenstorrent/tt-xla/issues/182)"
    )
)
def test_flax_roberta_large_inference(
    inference_tester: FlaxRobertaForMaskedLMTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_roberta_large_training(
    training_tester: FlaxRobertaForMaskedLMTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    training_tester.test()
