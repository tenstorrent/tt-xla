# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
from infra import RunMode
from utils import record_model_test_properties, runtime_fail

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
@pytest.mark.nightly
@pytest.mark.xfail(
    reason=(
        runtime_fail(
            "Host data with total size 16B does not match expected size 8B of device buffer! "
            "(https://github.com/tenstorrent/tt-xla/issues/182)"
        )
    )
)
def test_flax_bert_base_inference(
    inference_tester: FlaxBertForMaskedLMTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    inference_tester.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_bert_base_training(
    training_tester: FlaxBertForMaskedLMTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    training_tester.test()
