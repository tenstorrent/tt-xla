# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode

from ..tester import AlbertV2Tester

MODEL_PATH = "albert/albert-xlarge-v2"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> AlbertV2Tester:
    return AlbertV2Tester(MODEL_PATH)


@pytest.fixture
def training_tester() -> AlbertV2Tester:
    return AlbertV2Tester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.skip(reason="failed to legalize operation 'stablehlo.dot_general'")
def test_flax_albert_v2_xlarge_inference(
    inference_tester: AlbertV2Tester,
):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_albert_v2_xlarge_training(
    training_tester: AlbertV2Tester,
):
    training_tester.test()
