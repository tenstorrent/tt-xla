# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import runtime_fail

from ..tester import AlbertV2Tester

MODEL_PATH = "albert/albert-base-v2"
MODEL_NAME = "albert-v2-base"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> AlbertV2Tester:
    return AlbertV2Tester(MODEL_PATH)


@pytest.fixture
def training_tester() -> AlbertV2Tester:
    return AlbertV2Tester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.INFERENCE.value,
)
@pytest.mark.xfail(
    reason=(
        runtime_fail(
            "Atol comparison failed. Calculated: atol=131036.078125. Required: atol=0.16"
        )
    )
)
def test_flax_albert_v2_base_inference(inference_tester: AlbertV2Tester):
    inference_tester.test()


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.TRAINING.value,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_albert_v2_base_training(training_tester: AlbertV2Tester):
    training_tester.test()
