# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category, ExecutionPass, failed_runtime

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.longt5.text_classification.jax import (
    ModelLoader,
    ModelVariant,
)

from ..tester import LongT5Tester

VARIANT_NAME = ModelVariant.XL_TGLOBAL

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
@pytest.mark.large
def test_longt5_xl_tglobal_inference(inference_tester: LongT5Tester):
    inference_tester.test()


@pytest.mark.training
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
    execution_pass=ExecutionPass.FORWARD,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.xfail(
    reason=failed_runtime(
        "Out of Memory: Not enough space to allocate 204800000 B DRAM buffer across 12 banks, where each bank needs to store 17088000 B, but bank size is only 1073741792 B"
        "NO_ISSUE"
    )
)
@pytest.mark.large
def test_longt5_xl_tglobal_training(training_tester: LongT5Tester):
    training_tester.test()
