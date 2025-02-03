# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
from infra import RunMode
from utils import compile_fail, record_model_test_properties

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


@pytest.mark.xfail(
    reason=compile_fail(
        "Unsupported data type (https://github.com/tenstorrent/tt-xla/issues/214)"
    )
)
def test_flax_bart_large_inference(
    inference_tester: FlaxBartForCausalLMTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_bart_large_training(
    training_tester: FlaxBartForCausalLMTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    training_tester.test()
