# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import ModelTester, RunMode
from utils import compile_fail

from ..tester import BloomTester

MODEL_PATH = "bigscience/bloom-560m"
MODEL_NAME = "bloom-560m"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> BloomTester:
    return BloomTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> BloomTester:
    return BloomTester(ModelTester, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.nightly
@pytest.mark.record_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.INFERENCE.value,
)
@pytest.mark.skip(reason=compile_fail("Unsupported data type"))  # segfault
def test_bloom_560m_inference(inference_tester: BloomTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.TRAINING.value,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_bloom_560m_training(training_tester: BloomTester):
    training_tester.test()
