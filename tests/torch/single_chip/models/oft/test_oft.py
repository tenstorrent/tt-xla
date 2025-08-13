# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_ttmlir_compilation,
)

from .tester import OFTTester

VARIANT_NAME = "oft"

MODEL_NAME = build_model_name(
    Framework.TORCH,
    "oft",
    "base",
    ModelTask.CV_OBJECT_DET,
    ModelSource.CUSTOM,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> OFTTester:
    return OFTTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> OFTTester:
    return OFTTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "Broadcasting rule violation for rank >= 5 : dim -5, Broadcast is supported upto rank 4, dim a: 8, dim b: 1 "
        "https://github.com/tenstorrent/tt-xla/issues/821"
    )
)
def test_torch_oft_inference(inference_tester: OFTTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_oft_training(training_tester: OFTTester):
    training_tester.test()
