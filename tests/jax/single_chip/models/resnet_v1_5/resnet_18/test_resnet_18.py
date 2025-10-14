# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category, ExecutionPass, failed_ttmlir_compilation

from tests.infra.utilities.utils import create_jax_inference_tester
from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.resnet.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)

from ..tester import ResNetTester

VARIANT_NAME = ModelVariant.RESNET_18
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


def create_inference_tester(format: str) -> ResNetTester:
    """Create inference tester with specified format."""
    return create_jax_inference_tester(ResNetTester, VARIANT_NAME, format)


@pytest.fixture
def training_tester() -> ResNetTester:
    return ResNetTester(VARIANT_NAME, RunMode.TRAINING)


@pytest.fixture(
    params=[
        "float32",
        "bfloat16",
        pytest.param(
            "bfp8",
            marks=pytest.mark.skip(
                reason=(
                    "Skip until mixed-precision is supported in MLIR. https://github.com/tenstorrent/tt-mlir/issues/5252"
                )
            ),
        ),
    ],
    ids=str,  # test names will include the dtype string
)
def inference_tester(request) -> ResNetTester:
    tester = create_inference_tester(request.param)
    request.node.add_marker(pytest.mark.record_test_properties(dtype=request.param))
    return tester


@pytest.fixture
def training_tester() -> ResNetTester:
    return ResNetTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.PASSED,
)
def test_resnet_v1_5_18_inference(inference_tester: ResNetTester):
    inference_tester.test()


@pytest.mark.training
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
    execution_pass=ExecutionPass.BACKWARD,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "error: failed to legalize operation 'stablehlo.pad' "
        "https://github.com/tenstorrent/tt-mlir/issues/5305"
    )
)
def test_resnet_v1_5_18_training(training_tester: ResNetTester):
    training_tester.test()
