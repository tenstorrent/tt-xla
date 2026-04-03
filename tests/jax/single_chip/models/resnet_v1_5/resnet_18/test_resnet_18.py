# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from utils import BringupStatus, Category, failed_ttmlir_compilation

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


# ----- Tests -----


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.skip(
    reason=failed_ttmlir_compilation(
        "StableHLOLegalizeCompositePass assertion failure: "
        "'lowOp && low operand must be a ConstantOp' — "
        "TenstorrentUniformToRandConversionPattern cannot handle non-constant "
        "low operand in stablehlo.composite ops. "
        "https://github.com/tenstorrent/tt-xla/pull/4091"
    )
)
def test_resnet_v1_5_18_inference(inference_tester: ResNetTester):
    inference_tester.test()
