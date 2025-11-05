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
from third_party.tt_forge_models.regnet.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)

from ..tester import RegNetTester

VARIANT_NAME = ModelVariant.REGNET_Y_040
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def training_tester() -> RegNetTester:
    return RegNetTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


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
        "error: failed to legalize operation 'ttir.reverse' that was explicitly marked illegal "
        "https://github.com/tenstorrent/tt-mlir/issues/5663"
    )
)
def test_regnet_y_040_training(training_tester: RegNetTester):
    training_tester.test()
