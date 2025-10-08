# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from infra import RunMode
from utils import BringupStatus, Category, failed_runtime

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.vit.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)

from ..tester import ViTTester

VARIANT_NAME = ModelVariant.BASE_PATCH32_384
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> ViTTester:
    return ViTTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> ViTTester:
    return ViTTester(VARIANT_NAME, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.xfail(
    reason=failed_runtime(
        "Out of Memory: Not enough space to allocate  7782400 B L1 buffer across 5 banks, "
        "where each bank needs to store 1556480 B "
        "(https://github.com/tenstorrent/tt-xla/issues/918)"
    )
)
def test_vit_base_patch32_384_inference(
    inference_tester: ViTTester,
):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_vit_base_patch32_384_training(training_tester: ViTTester):
    training_tester.test()
