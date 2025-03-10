# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
from infra import RunMode
from utils import compile_fail, record_model_test_properties

from ..tester import FlaxBeitForImageClassificationTester

MODEL_PATH = "microsoft/beit-large-patch16-224"
MODEL_NAME = "beit-large"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxBeitForImageClassificationTester:
    return FlaxBeitForImageClassificationTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> FlaxBeitForImageClassificationTester:
    return FlaxBeitForImageClassificationTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.xfail(reason=compile_fail("failed to legalize operation 'ttir.gather'"))
def test_flax_beit_large_inference(
    inference_tester: FlaxBeitForImageClassificationTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    inference_tester.test()


@pytest.mark.model_test
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_beit_large_training(
    training_tester: FlaxBeitForImageClassificationTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    training_tester.test()
