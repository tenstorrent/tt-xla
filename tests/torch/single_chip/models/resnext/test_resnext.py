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
from third_party.tt_forge_models.resnext.pytorch import ModelVariant
from .tester import ResnextTester

VARIANT_NAME = ModelVariant.RESNEXT50_32X4D


MODEL_NAME = build_model_name(
    Framework.TORCH,
    "resnext",
    "_32x4d",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.TORCH_HUB,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> ResnextTester:
    return ResnextTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> ResnextTester:
    return ResnextTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_torch_resnext_inference(inference_tester: ResnextTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_resnext_training(training_tester: ResnextTester):
    training_tester.test()
