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
    incorrect_result,
)

from .tester import ResnetTester
from third_party.tt_forge_models.resnet.pytorch import ModelVariant
from tests.infra.testers.compiler_config import CompilerConfig

VARIANT_NAME = ModelVariant.RESNET_50_HF


MODEL_NAME = build_model_name(
    Framework.TORCH,
    "resnet",
    "50",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.TORCH_HUB,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> ResnetTester:
    compiler_config = CompilerConfig(enable_optimizer=True)
    return ResnetTester(VARIANT_NAME, compiler_config=compiler_config)


@pytest.fixture
def training_tester() -> ResnetTester:
    return ResnetTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
@pytest.mark.xfail(
    reason=incorrect_result(
        "PCC comparison failed. Calculated: pcc=nan. Required: pcc=0.99 "
        "https://github.com/tenstorrent/tt-xla/issues/1384"
    )
)
def test_torch_resnet_inference(inference_tester: ResnetTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_resnet_training(training_tester: ResnetTester):
    training_tester.test()
