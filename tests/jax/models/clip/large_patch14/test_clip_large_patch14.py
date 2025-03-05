# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import compile_fail

from ..tester import FlaxCLIPTester

MODEL_PATH = "openai/clip-vit-large-patch14"
MODEL_NAME = "clip-large-patch14"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxCLIPTester:
    return FlaxCLIPTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> FlaxCLIPTester:
    return FlaxCLIPTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.nightly
@pytest.mark.record_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.INFERENCE.value,
)
@pytest.mark.skip(
    reason=compile_fail(
        'Assertion `llvm::isUIntN(BitWidth, val) && "Value is not an N-bit unsigned value"\' failed.'
    )
)
def test_clip_large_patch14_inference(inference_tester: FlaxCLIPTester):
    inference_tester.test()


@pytest.mark.record_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.TRAINING.value,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_clip_large_patch14_training(training_tester: FlaxCLIPTester):
    training_tester.test()
