# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
from infra import ModelTester, RunMode
from utils import record_model_test_properties

from ..tester import GPTNeoTester

MODEL_PATH = "EleutherAI/gpt-neo-2.7b"
MODEL_NAME = "gpt-neo-2.7b"

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> GPTNeoTester:
    return GPTNeoTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> GPTNeoTester:
    return GPTNeoTester(ModelTester, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.nightly
@pytest.mark.skip(reason="OOMs on CI.")
def test_gpt_neo_2_7b_inference(
    inference_tester: GPTNeoTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.skip(reason="Support for training not implemented")
def test_gpt_neo_2_7b_training(
    training_tester: GPTNeoTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    training_tester.test()
