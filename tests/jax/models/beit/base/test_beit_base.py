# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import compile_fail

from ..tester import FlaxBeitForImageClassificationTester

MODEL_PATH = "microsoft/beit-base-patch16-224"
MODEL_NAME = "beit-base"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxBeitForImageClassificationTester:
    return FlaxBeitForImageClassificationTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> FlaxBeitForImageClassificationTester:
    return FlaxBeitForImageClassificationTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.INFERENCE.value,
)
@pytest.mark.xfail(reason=compile_fail("failed to legalize operation 'ttir.gather'"))
def test_flax_beit_base_inference(
    inference_tester: FlaxBeitForImageClassificationTester,
):
    inference_tester.test()


@pytest.mark.model_test
@pytest.mark.record_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.TRAINING.value,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_beit_base_training(training_tester: FlaxBeitForImageClassificationTester):
    training_tester.test()
