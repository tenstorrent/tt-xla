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

from third_party.tt_forge_models.clip.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)
from third_party.tt_forge_models.config import Parallelism

from ..tester import FlaxCLIPTester

VARIANT_NAME = ModelVariant.LARGE_PATCH14

MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxCLIPTester:
    return FlaxCLIPTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> FlaxCLIPTester:
    return FlaxCLIPTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


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
        "PCC comparison failed. Calculated: pcc=-1.0. Required: pcc=0.99 "
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
def test_clip_large_patch14_inference(inference_tester: FlaxCLIPTester):
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
        "error: failed to legalize operation 'ttir.scatter'"
        "https://github.com/tenstorrent/tt-mlir/issues/4792"
    )
)
def test_clip_large_patch14_training(training_tester: FlaxCLIPTester):
    training_tester.test()
