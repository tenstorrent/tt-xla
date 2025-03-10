# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from infra import RunMode
from utils import runtime_fail

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


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.INFERENCE.value,
)
@pytest.mark.skip(
    reason=runtime_fail(
        "Host data with total size 4B does not match expected size 2B of device buffer!"
    )
)
def test_gpt_neo_125m_inference(inference_tester: GPTNeoTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.TRAINING.value,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_gpt_neo_125m_training(training_tester: GPTNeoTester):
    training_tester.test()
