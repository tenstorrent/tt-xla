# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
from infra import ModelTester, RunMode
from utils import record_model_test_properties, runtime_fail

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
@pytest.mark.xfail(
    reason=runtime_fail(
        "Cannot get the device from a tensor with host storage "
        "(https://github.com/tenstorrent/tt-xla/issues/171)"
    )
)
def test_gpt2_base_inference(
    inference_tester: GPT2Tester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    inference_tester.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.skip(reason="Support for training not implemented")
def test_gpt2_base_training(
    training_tester: GPT2Tester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    training_tester.test()
