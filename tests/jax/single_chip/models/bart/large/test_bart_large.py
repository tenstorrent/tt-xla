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

from third_party.tt_forge_models.bart.causal_lm.jax import ModelLoader, ModelVariant
from third_party.tt_forge_models.config import Parallelism

from ..tester import FlaxBartForCausalLMTester

VARIANT_NAME = ModelVariant.LARGE

MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def training_tester() -> FlaxBartForCausalLMTester:
    return FlaxBartForCausalLMTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.xfail(
    reason=incorrect_result(
        "PCC comparison failed. Calculated: pcc=0.9883341789245605. Required: pcc=0.99."
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
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
def test_flax_bart_large_training(training_tester: FlaxBartForCausalLMTester):
    training_tester.test()
