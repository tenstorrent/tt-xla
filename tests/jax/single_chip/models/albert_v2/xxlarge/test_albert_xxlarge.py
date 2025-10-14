# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
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

from third_party.tt_forge_models.albert.masked_lm.jax import ModelLoader, ModelVariant
from third_party.tt_forge_models.config import Parallelism

from ..tester import AlbertV2Tester

VARIANT_NAME = ModelVariant.XXLARGE_V2

MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> AlbertV2Tester:
    return AlbertV2Tester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> AlbertV2Tester:
    return AlbertV2Tester(VARIANT_NAME, run_mode=RunMode.TRAINING)


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
        "PCC comparison failed. Calculated: pcc=0.9695068001747131. Required: pcc=0.99 "
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
def test_flax_albert_v2_xxlarge_inference(inference_tester: AlbertV2Tester):
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
        "error: failed to legalize operation 'ttir.scatter' "
        "https://github.com/tenstorrent/tt-mlir/issues/5091"
    )
)
def test_flax_albert_v2_xxlarge_training(training_tester: AlbertV2Tester):
    training_tester.test()
