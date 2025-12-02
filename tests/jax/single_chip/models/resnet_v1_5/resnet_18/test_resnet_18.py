# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.infra.utilities.utils import create_jax_inference_tester
from third_party.tt_forge_models.resnet.image_classification.jax import ModelVariant

from ..tester import ResNetTester

VARIANT_NAME = ModelVariant.RESNET_18

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
