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

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.t5.summarization.jax import ModelLoader, ModelVariant

from ..tester import T5Tester

VARIANT_NAME = ModelVariant.SMALL
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> T5Tester:
    return T5Tester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> T5Tester:
    return T5Tester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
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
        "PCC comparison failed on Blackhole. Calculated: pcc=0.7849299907684326. Required: pcc=0.99."
        "https://github.com/tenstorrent/tt-xla/issues/1038"
    )
)
def test_t5_small_inference(inference_tester: T5Tester):
    inference_tester.test()


@pytest.mark.push
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
def test_t5_small_training(training_tester: T5Tester):
    training_tester.test()
