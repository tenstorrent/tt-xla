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
def inference_tester() -> FlaxBeitForImageClassificationTester:
    return FlaxBeitForImageClassificationTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> FlaxBeitForImageClassificationTester:
    return FlaxBeitForImageClassificationTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
@pytest.mark.xfail(
    reason=incorrect_result(
        "PCC comparison failed. Calculated: pcc0.09154801815748215. Required: pcc=0.99 "
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
def test_flax_beit_large_inference(
    inference_tester: FlaxBeitForImageClassificationTester,
):
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
        "error: 'ttir.conv2d' op The output tensor height and width dimension (224, 224) do not match the expected dimensions (29, 29) "
        "https://github.com/tenstorrent/tt-mlir/issues/5304"
    )
)
def test_flax_beit_large_training(
    training_tester: FlaxBeitForImageClassificationTester,
):
    training_tester.test()
