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
    failed_runtime,
)
from third_party.tt_forge_models.vit.image_classification.jax import ModelVariant
from ..tester import ViTTester

VARIANT_NAME = ModelVariant.LARGE_PATCH32_224_IN_21K
MODEL_NAME = build_model_name(
    Framework.JAX,
    "vit",
    str(VARIANT_NAME),
    ModelTask.CV_IMAGE_CLS,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> ViTTester:
    return ViTTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> ViTTester:
    return ViTTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.xfail(
    reason=failed_runtime(
        "Out of Memory: Not enough space to allocate  2287616 B L1 buffer across 2 banks, "
        "where each bank needs to store 1143808 B "
        "(https://github.com/tenstorrent/tt-xla/issues/918)"
    )
)
def test_vit_large_patch32_224_in21k_inference(
    inference_tester: ViTTester,
):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_vit_large_patch32_224_in21k_training(training_tester: ViTTester):
    training_tester.test()
