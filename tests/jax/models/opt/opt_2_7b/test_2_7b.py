# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import compile_fail

from ..tester import OPTTester

MODEL_PATH = "facebook/opt-2.7b"
MODEL_NAME = "opt-2.7b"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> OPTTester:
    return OPTTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> OPTTester:
    return OPTTester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.INFERENCE.value,
)
@pytest.mark.skip(reason=compile_fail("Unsupported data type"))  # segfault
def test_opt_2_7b_inference(inference_tester: OPTTester):
    inference_tester.test()


@pytest.mark.model_test
@pytest.mark.record_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.TRAINING.value,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_opt_2_7b_training(training_tester: OPTTester):
    training_tester.test()
