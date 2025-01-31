# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from infra import ModelTester, RunMode

from ..tester import GPT2Tester

MODEL_PATH = "openai-community/gpt2-large"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> GPT2Tester:
    return GPT2Tester(MODEL_PATH)


@pytest.fixture
def training_tester() -> GPT2Tester:
    return GPT2Tester(ModelTester, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.xfail(
    reason="Cannot get the device from a tensor with host storage (https://github.com/tenstorrent/tt-xla/issues/171)"
)
def test_gpt2_large_inference(
    inference_tester: GPT2Tester,
):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_gpt2_large_training(
    training_tester: GPT2Tester,
):
    training_tester.test()
