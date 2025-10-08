# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil

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

from .tester import Monodepth2Tester

VARIANT_NAME = "mono_640x192"

MODEL_NAME = build_model_name(
    Framework.TORCH,
    "monodepth2",
    "base",
    ModelTask.CV_DEPTH_EST,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> Monodepth2Tester:
    return Monodepth2Tester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> Monodepth2Tester:
    return Monodepth2Tester(VARIANT_NAME, run_mode=RunMode.TRAINING)


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
        "failed to legalize operation 'ttir.reverse' "
        "https://github.com/tenstorrent/tt-xla/issues/736"
    )
)
def test_torch_monodepth2_inference(inference_tester: Monodepth2Tester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_monodepth2_training(training_tester: Monodepth2Tester):
    training_tester.test()
