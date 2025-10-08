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
)

from ..tester import OPTTester
from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.opt.causal_lm.jax import ModelVariant, ModelLoader

VARIANT_NAME = ModelVariant._1_3B
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> OPTTester:
    return OPTTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> OPTTester:
    return OPTTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_opt_1_3b_inference(inference_tester: OPTTester):
    inference_tester.test()


@pytest.mark.nightly
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
        "error: failed to legalize operation 'ttir.scatter'"
        "https://github.com/tenstorrent/tt-mlir/issues/4792"
    )
)
def test_opt_1_3b_training(training_tester: OPTTester):
    training_tester.test()
