# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import runtime_fail

from ..tester import AlbertV2Tester

MODEL_PATH = "albert/albert-xlarge-v2"
MODEL_NAME = "albert-v2-xlarge"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> AlbertV2Tester:
    return AlbertV2Tester(MODEL_PATH)


@pytest.fixture
def training_tester() -> AlbertV2Tester:
    return AlbertV2Tester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.nightly
@pytest.mark.record_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.INFERENCE.value,
)
@pytest.mark.xfail(
    reason=(
        runtime_fail(
            "Host data with total size 24B does not match expected size 12B of device buffer! "
            "(https://github.com/tenstorrent/tt-xla/issues/182)"
        )
    )
)
def test_flax_albert_v2_xlarge_inference(inference_tester: AlbertV2Tester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.TRAINING.value,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_albert_v2_xlarge_training(training_tester: AlbertV2Tester):
    training_tester.test()
