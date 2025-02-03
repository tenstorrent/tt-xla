# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
from infra import RunMode
from utils import record_model_test_properties

from ..tester import LLamaTester

MODEL_PATH = "openlm-research/open_llama_3b_v2"
MODEL_NAME = "open-llama-3b-v2"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> LLamaTester:
    return LLamaTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> LLamaTester:
    return LLamaTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.skip(
    reason="OOMs in CI (https://github.com/tenstorrent/tt-xla/issues/186)"
)
def test_openllama3b_inference(
    inference_tester: LLamaTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_openllama3b_training(
    training_tester: LLamaTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    training_tester.test()
