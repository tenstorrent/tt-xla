# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import runtime_fail

from ..tester import FlaxBartForCausalLMTester

MODEL_PATH = "facebook/bart-large"
MODEL_NAME = "bart-large"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxBartForCausalLMTester:
    return FlaxBartForCausalLMTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> FlaxBartForCausalLMTester:
    return FlaxBartForCausalLMTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.INFERENCE.value,
)
@pytest.mark.xfail(
    reason=(
        runtime_fail(
            "Invalid arguments to reshape "
            "(https://github.com/tenstorrent/tt-xla/issues/307)"
        )
    )
)
def test_flax_bart_large_inference(inference_tester: FlaxBartForCausalLMTester):
    inference_tester.test()


@pytest.mark.model_test
@pytest.mark.record_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.TRAINING.value,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_bart_large_training(training_tester: FlaxBartForCausalLMTester):
    training_tester.test()
