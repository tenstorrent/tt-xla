# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category, ExecutionPass, failed_ttmlir_compilation

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.electra.causal_lm.jax import ModelLoader, ModelVariant

from ..tester import ElectraTester

VARIANT_NAME = ModelVariant.LARGE_DISCRIMINATOR

MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> ElectraTester:
    return ElectraTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> ElectraTester:
    return ElectraTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.PASSED,
)
def test_electra_large_discriminator_inference(inference_tester: ElectraTester):
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
        "https://github.com/tenstorrent/tt-mlir/issues/4792"
    )
)
def test_electra_large_discriminator_training(training_tester: ElectraTester):
    training_tester.test()
