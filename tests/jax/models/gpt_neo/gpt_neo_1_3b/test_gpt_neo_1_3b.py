# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import runtime_fail

from ..tester import GPTNeoTester

MODEL_PATH = "EleutherAI/gpt-neo-1.3b"
MODEL_NAME = "gpt-neo-1.3b"

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> GPTNeoTester:
    return GPTNeoTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> GPTNeoTester:
    return GPTNeoTester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.skip(reason=runtime_fail("OOMs in CI"))
def test_gpt_neo_1_3b_inference(inference_tester: GPTNeoTester):
    inference_tester.test()


@pytest.mark.model_test
@pytest.mark.skip(reason="Support for training not implemented")
def test_gpt_neo_1_3b_training(training_tester: GPTNeoTester):
    training_tester.test()
