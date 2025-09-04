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

from third_party.tt_forge_models.centernet.pytorch import ModelVariant
from .tester import CenterNetTester

VARIANT_NAME = ModelVariant.HOURGLASS_COCO


MODEL_NAME = build_model_name(
    Framework.TORCH,
    "centernet",
    "hourglass_coco",
    ModelTask.CV_OBJECT_DET,
    ModelSource.TORCH_HUB,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> CenterNetTester:
    return CenterNetTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> CenterNetTester:
    return CenterNetTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
def test_torch_centernet_inference(inference_tester: CenterNetTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_centernet_training(training_tester: CenterNetTester):
    training_tester.test()
