# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from infra import RunMode
from utils import BringupStatus, Category, ExecutionPass, failed_ttmlir_compilation

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.vit.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)

from ..tester import ViTTester

VARIANT_NAME = ModelVariant.HUGE_PATCH14_224_IN_21K
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)


# ----- Fixtures -----


@pytest.fixture
def training_tester() -> ViTTester:
    return ViTTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.test_forge_models_training
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.TRAINING,
    execution_pass=ExecutionPass.BACKWARD,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "Invalid data size. numElements * elementSize == data->size() "
        "https://github.com/tenstorrent/tt-mlir/issues/5665"
    )
)
def test_vit_huge_patch14_224_in21k_training(training_tester: ViTTester):
    training_tester.test()
