# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
from infra import RunMode
from utils import record_model_test_properties, runtime_fail

from ..tester import GPTNeoTester

MODEL_PATH = "EleutherAI/gpt-neo-125m"
MODEL_NAME = "gpt-neo-125m"

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> GPTNeoTester:
    return GPTNeoTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> GPTNeoTester:
    return GPTNeoTester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.xfail(
    reason=runtime_fail(
        "Host data with total size 4B does not match expected size 2B of device buffer!"
    )
)
def test_gpt_neo_125m_inference(
    inference_tester: GPTNeoTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    inference_tester.test()


@pytest.mark.model_test
@pytest.mark.skip(reason="Support for training not implemented")
def test_gpt_neo_125m_training(
    training_tester: GPTNeoTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    training_tester.test()
