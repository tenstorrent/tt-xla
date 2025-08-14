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
)

from .tester import OpenposeTester

VARIANT_NAME = "lwopenpose2d_mobilenet_cmupan_coco"

MODEL_NAME = build_model_name(
    Framework.TORCH,
    "openpose",
    "base",
    ModelTask.CV_KEYPOINT_DET,
    ModelSource.CUSTOM,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> OpenposeTester:
    return OpenposeTester(VARIANT_NAME)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_torch_openpose_inference(inference_tester: OpenposeTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_openpose_training(training_tester: OpenposeTester):
    training_tester.test()
