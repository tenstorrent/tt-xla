# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode

from ..tester import FlaxBartForCausalLMTester

MODEL_PATH = "facebook/bart-base"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxBartForCausalLMTester:
    return FlaxBartForCausalLMTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> FlaxBartForCausalLMTester:
    return FlaxBartForCausalLMTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.xfail(reason="failed to legalize operation 'stablehlo.reduce'")
def test_flax_bart_base_inference(
    inference_tester: FlaxBartForCausalLMTester,
):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_bart_base_training(
    training_tester: FlaxBartForCausalLMTester,
):
    training_tester.test()
