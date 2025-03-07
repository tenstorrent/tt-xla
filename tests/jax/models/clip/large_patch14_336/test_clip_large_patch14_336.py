# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


from typing import Callable

import pytest
from infra import RunMode
from utils import compile_fail, record_model_test_properties

from ..tester import FlaxCLIPTester

MODEL_PATH = "openai/clip-vit-large-patch14-336"
MODEL_NAME = "clip-large-patch14-336"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxCLIPTester:
    return FlaxCLIPTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> FlaxCLIPTester:
    return FlaxCLIPTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.xfail(
    reason=compile_fail("failed to legalize operation 'stablehlo.reduce'")
)
def test_clip_large_patch14_336_inference(
    inference_tester: FlaxCLIPTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_clip_large_patch14_336_training(
    training_tester: FlaxCLIPTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    training_tester.test()
