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


def create_inference_tester(format: str, optimization_level: int) -> ResnetTester:
    """Create inference tester with specified format and optimization level."""
    compiler_config = CompilerConfig(
        optimization_level=optimization_level,
    )
    return create_torch_inference_tester(
        ResnetTester, VARIANT_NAME, format, compiler_config=compiler_config
    )


@pytest.fixture
def trace_tester(monkeypatch: MonkeyPatch) -> ResnetTester:
    monkeypatch.setenv("TT_RUNTIME_TRACE_REGION_SIZE", "10000000")

    compiler_config = CompilerConfig(optimization_level=1, enable_trace=True)
    return ResnetTester(VARIANT_NAME, compiler_config=compiler_config)


@pytest.fixture
def training_tester() -> ResnetTester:
    return ResnetTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
@pytest.mark.parametrize(
    "format,optimization_level",
    [
        pytest.param(
            "bfp8",
            1,
            marks=pytest.mark.xfail(
                reason="bfp8 with optimization_level_1 has hudge data mismatch. Tracking issue: https://github.com/tenstorrent/tt-xla/issues/1673"
            ),
        ),
        pytest.param(
            "bfp8",
            0,
            marks=pytest.mark.xfail(
                reason="ttnn.maximum not supported for bfp8. This op should be fused to ReLU. Tracking mlir issue: https://github.com/tenstorrent/tt-mlir/issues/5329 "
            ),
        ),
        pytest.param(
            "bfloat16",
            1,
            marks=pytest.mark.xfail(
                reason="PCC comparison < 0.99 (observed ~0.982-0.984)"
            ),
        ),
        pytest.param(
            "bfloat16",
            0,
            marks=pytest.mark.xfail(
                reason="PCC comparison < 0.99 (observed ~0.982-0.984)"
            ),
        ),
        pytest.param("float32", 1),
        pytest.param(
            "float32",
            0,
            marks=pytest.mark.xfail(
                reason="PCC comparison < 0.99 (observed ~0.9875). Small, mentioned here anyways: https://github.com/tenstorrent/tt-xla/issues/1673"
            ),
        ),
    ],
    ids=[
        "optimization_level_1-bfp8",
        "optimization_level_0-bfp8",
        "optimization_level_1-bfloat16",
        "optimization_level_0-bfloat16",
        "optimization_level_1-float32",
        "optimization_level_0-float32",
    ],
)
def test_torch_resnet_inference(format: str, optimization_level: int):
    tester = create_inference_tester(format, optimization_level)
    tester.test()


@pytest.mark.push
@pytest.mark.single_device
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
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_resnet_training(training_tester: ResnetTester):
    training_tester.test()
