# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category, failed_runtime

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.longt5.text_classification.jax import (
    ModelLoader,
    ModelVariant,
)

from ..tester import LongT5Tester

VARIANT_NAME = ModelVariant.LARGE_LOCAL

MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> LongT5Tester:
    return LongT5Tester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> LongT5Tester:
    return LongT5Tester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.xfail(
    reason=failed_runtime(
        "ttnn::pad only supports padding on the lowest 3 dimensions for tensors with rank > 4 1 "
        "https://github.com/tenstorrent/tt-xla/issues/580"
    )
)
def test_longt5_large_local_inference(inference_tester: LongT5Tester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_longt5_large_local_training(training_tester: LongT5Tester):
    training_tester.test()
