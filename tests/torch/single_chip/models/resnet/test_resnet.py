# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from pytest import MonkeyPatch
from utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    incorrect_result,
)

from tests.infra.testers.compiler_config import CompilerConfig
from tests.infra.utilities.utils import create_torch_inference_tester
from third_party.tt_forge_models.resnet.pytorch import ModelVariant

from .tester import ResnetTester

VARIANT_NAME = ModelVariant.RESNET_50_HF


MODEL_NAME = build_model_name(
    Framework.TORCH,
    "resnet",
    "50",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.TORCH_HUB,
)


# ----- Fixtures -----


def create_inference_tester(format: str, enable_optimizer: bool) -> ResnetTester:
    """Create inference tester with specified format and optimizer settings."""
    compiler_config = CompilerConfig(
        enable_optimizer=enable_optimizer,
        enable_fusing_conv2d_with_multiply_pattern=True,
    )
    return create_torch_inference_tester(
        ResnetTester, VARIANT_NAME, format, compiler_config=compiler_config
    )


@pytest.fixture
def trace_tester(monkeypatch: MonkeyPatch) -> ResnetTester:
    monkeypatch.setenv("TT_RUNTIME_ENABLE_PROGRAM_CACHE", "1")
    monkeypatch.setenv("TT_RUNTIME_TRACE_REGION_SIZE", "10000000")

    compiler_config = CompilerConfig(enable_optimizer=True, enable_trace=True)
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
@pytest.mark.parametrize(
    "format,enable_optimizer",
    [
        pytest.param(
            "bfp8",
            True,
            marks=pytest.mark.xfail(
                reason="bfp8 with optimizer_on has hudge data mismatch. Tracking issue: https://github.com/tenstorrent/tt-xla/issues/1673"
            ),
        ),
        pytest.param(
            "bfp8",
            False,
            marks=pytest.mark.xfail(
                reason="ttnn.maximum not supported for bfp8. This op should be fused to ReLU. Tracking mlir issue: https://github.com/tenstorrent/tt-mlir/issues/5329 "
            ),
        ),
        pytest.param(
            "bfloat16",
            True,
            marks=pytest.mark.xfail(
                reason="PCC comparison < 0.99 (observed ~0.982-0.984)"
            ),
        ),
        pytest.param(
            "bfloat16",
            False,
            marks=pytest.mark.xfail(
                reason="PCC comparison < 0.99 (observed ~0.982-0.984)"
            ),
        ),
        pytest.param("float32", True),
        pytest.param("float32", False),
    ],
    ids=[
        "optimizer_on-bfp8",
        "optimizer_off-bfp8",
        "optimizer_on-bfloat16",
        "optimizer_off-bfloat16",
        "optimizer_on-float32",
        "optimizer_off-float32",
    ],
)
def test_torch_resnet_inference(format: str, enable_optimizer: bool):
    tester = create_inference_tester(format, enable_optimizer)
    tester.test()


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
def test_torch_resnet_inference_trace(trace_tester: ResnetTester):
    trace_tester.test()


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
