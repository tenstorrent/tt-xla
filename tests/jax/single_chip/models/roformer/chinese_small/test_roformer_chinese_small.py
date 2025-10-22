# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category, ExecutionPass, failed_ttmlir_compilation

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.roformer.masked_lm.jax import ModelLoader, ModelVariant

from ..tester import RoFormerTester

VARIANT_NAME = ModelVariant.CHINESE_SMALL
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def training_tester() -> RoFormerTester:
    return RoFormerTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.training
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
        "error: failed to legalize operation 'ttir.scatter' "
        "https://github.com/tenstorrent/tt-mlir/issues/4792"
    )
)
def test_roformer_chinese_small_training(training_tester: RoFormerTester):
    training_tester.test()
