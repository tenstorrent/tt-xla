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

from ..tester import ResNetTester
from third_party.tt_forge_models.resnet.image_classification.jax import ModelVariant
from tests.infra.utilities.utils import create_jax_inference_tester

VARIANT_NAME = ModelVariant.RESNET_18
MODEL_NAME = build_model_name(
    Framework.JAX,
    "resnet_v1.5",
    "18",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


def create_inference_tester(format: str) -> ResNetTester:
    """Create inference tester with specified format."""
    return create_jax_inference_tester(ResNetTester, VARIANT_NAME, format)


@pytest.fixture
def training_tester() -> ResNetTester:
    return ResNetTester(VARIANT_NAME, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
@pytest.mark.parametrize(
    "format",
    [
        "float32",
        "bfloat16",
        pytest.param(
            "bfp8",
            marks=pytest.mark.skip(
                reason="Encountered mlir error: Expected TileType for non-bfloat16 element type. Currently MLIR bfp8 pass requires all weights to be in bfloat16 and stablehlo contains cast to f32 for reduction used in softmax."
            ),
        ),
    ],
)
def test_resnet_v1_5_18_inference(format: str):
    tester = create_inference_tester(format)
    tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_resnet_v1_5_18_training(training_tester: ResNetTester):
    training_tester.test()
