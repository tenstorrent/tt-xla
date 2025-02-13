# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
from infra import ModelTester, RunMode
from utils import record_model_test_properties, runtime_fail

from ..tester import GPT2Tester

MODEL_PATH = "openai-community/gpt2-medium"
MODEL_NAME = "gpt2-medium"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> GPT2Tester:
    return GPT2Tester(MODEL_PATH)


@pytest.fixture
def training_tester() -> GPT2Tester:
    return GPT2Tester(ModelTester, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.nightly
@pytest.mark.xfail(
    reason=runtime_fail(
        "Host data with total size 4B does not match expected size 2B of device buffer! "
        "(https://github.com/tenstorrent/tt-xla/issues/182)"
    )
)
def test_gpt2_medium_inference(
    inference_tester: GPT2Tester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.skip(reason="Support for training not implemented")
def test_gpt2_medium_training(
    training_tester: GPT2Tester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    training_tester.test()
