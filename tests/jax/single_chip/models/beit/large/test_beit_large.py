# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import (
    BringupStatus,
    Category,
    ExecutionPass,
    failed_ttmlir_compilation,
    incorrect_result,
)

from third_party.tt_forge_models.beit.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)
from third_party.tt_forge_models.config import Parallelism

from ..tester import FlaxBeitForImageClassificationTester

VARIANT_NAME = ModelVariant.LARGE

MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def training_tester() -> FlaxBeitForImageClassificationTester:
    return FlaxBeitForImageClassificationTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.test_forge_models_training
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
        "Invalid data size. numElements * elementSize == data->size() "
        "https://github.com/tenstorrent/tt-mlir/issues/5665"
    )
)
def test_flax_beit_large_training(
    training_tester: FlaxBeitForImageClassificationTester,
):
    training_tester.test()
