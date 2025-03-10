# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import ModelTester, RunMode
from utils import runtime_fail

from ..tester import GPT2Tester

MODEL_PATH = "openai-community/gpt2"
MODEL_NAME = "gpt2-base"

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> GPT2Tester:
    return GPT2Tester(MODEL_PATH)


@pytest.fixture
def training_tester() -> GPT2Tester:
    return GPT2Tester(ModelTester, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.INFERENCE.value,
)
@pytest.mark.xfail(
    reason=runtime_fail(
        "Host data with total size 4B does not match expected size 2B of device buffer! "
        "(https://github.com/tenstorrent/tt-xla/issues/182)"
    )
)
def test_gpt2_base_inference(inference_tester: GPT2Tester):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.TRAINING.value,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_gpt2_base_training(training_tester: GPT2Tester):
    training_tester.test()
