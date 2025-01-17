# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
from infra import RunMode
from utils import record_model_test_properties, runtime_fail

from ..tester import AlbertV2Tester

MODEL_PATH = "albert/albert-base-v2"
MODEL_NAME = MODEL_PATH.split("/")[1]


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> AlbertV2Tester:
    return AlbertV2Tester(MODEL_PATH)


@pytest.fixture
def training_tester() -> AlbertV2Tester:
    return AlbertV2Tester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.xfail(
    reason=(
        runtime_fail(
            "Cannot get the device from a tensor with host storage "
            "(https://github.com/tenstorrent/tt-xla/issues/171)"
        )
    )
)
def test_flax_albert_v2_base_inference(
    inference_tester: AlbertV2Tester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_albert_v2_base_training(
    training_tester: AlbertV2Tester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    training_tester.test()
