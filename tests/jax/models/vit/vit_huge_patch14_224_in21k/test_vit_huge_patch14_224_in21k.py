# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
from infra import RunMode
from utils import record_model_test_properties, runtime_fail

from ..tester import ViTTester

MODEL_PATH = "google/vit-huge-patch14-224-in21k"
MODEL_NAME = "vit-huge-patch14-224-in21k"


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> ViTTester:
    return ViTTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> ViTTester:
    return ViTTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.xfail(
    reason=runtime_fail(
        "Out of memory while performing convolution."
        "(https://github.com/tenstorrent/tt-xla/issues/187)"
    )
)
def test_vit_huge_patch14_224_in21k_inference(
    inference_tester: ViTTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)
    inference_tester.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.skip(reason="Support for training not implemented")
def test_vit_huge_patch14_224_in21k_training(
    training_tester: ViTTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)
    training_tester.test()
