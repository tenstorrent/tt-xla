# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from pytest import MonkeyPatch
from utils import (
    BringupStatus,
    Category,
    ExecutionPass,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_ttmlir_compilation,
    incorrect_result,
)

from third_party.tt_forge_models.resnet.image_classification.jax import ModelVariant

from ..tester import CompilerConfig, ResNetTester

VARIANT_NAME = ModelVariant.RESNET_50
MODEL_NAME = build_model_name(
    Framework.JAX,
    "resnet_v1.5",
    "50",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> ResNetTester:
    return ResNetTester(VARIANT_NAME)


@pytest.fixture
def trace_tester(monkeypatch: MonkeyPatch) -> ResNetTester:
    # These need to be set before the tester is created
    monkeypatch.setenv("TT_RUNTIME_ENABLE_PROGRAM_CACHE", "1")
    monkeypatch.setenv("TT_RUNTIME_TRACE_REGION_SIZE", "10000000")

    cc = CompilerConfig(enable_optimizer=True, enable_trace=True)
    return ResNetTester(VARIANT_NAME, compiler_config=cc)


@pytest.fixture
def training_tester() -> ResNetTester:
    return ResNetTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
@pytest.mark.large
def test_resnet_v1_5_50_inference(inference_tester: ResNetTester):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
@pytest.mark.large
def test_resnet_v1_5_50_inference_trace(
    trace_tester: ResNetTester,
):
    trace_tester.test()


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.RED,
    run_mode=RunMode.TRAINING,
    execution_pass=ExecutionPass.BACKWARD,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.large
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "error: failed to legalize operation 'stablehlo.pad'" "NO_ISSUE"
    )
)
def test_resnet_v1_5_50_training(training_tester: ResNetTester):
    training_tester.test()
