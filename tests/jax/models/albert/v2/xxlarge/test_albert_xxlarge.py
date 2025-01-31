# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode

from ..tester import AlbertV2Tester

MODEL_PATH = "albert/albert-xxlarge-v2"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> AlbertV2Tester:
    return AlbertV2Tester(MODEL_PATH)


@pytest.fixture
def training_tester() -> AlbertV2Tester:
    return AlbertV2Tester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.xfail(
    reason="Cannot get the device from a tensor with host storage (https://github.com/tenstorrent/tt-xla/issues/171)"
)
def test_flax_albert_v2_xxlarge_inference(
    inference_tester: AlbertV2Tester,
):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_albert_v2_xxlarge_training(
    training_tester: AlbertV2Tester,
):
    training_tester.test()
