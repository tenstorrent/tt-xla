# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Callable

import pytest
from infra import RunMode
from utils import compile_fail, record_model_test_properties

from ..tester import ResNetTester

MODEL_PATH = "microsoft/resnet-50"
MODEL_NAME = "resnet-50"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> ResNetTester:
    return ResNetTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> ResNetTester:
    return ResNetTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.nightly
@pytest.mark.xfail(
    reason=compile_fail("failed to legalize operation 'stablehlo.reduce_window'")
)
def test_resnet_50_inference(
    inference_tester: ResNetTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.skip(reason="Support for training not implemented")
def test_resnet_50_training(
    training_tester: ResNetTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    training_tester.test()
