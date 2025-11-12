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
from third_party.tt_forge_models.resnet.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)

from ..tester import ResNetTester

VARIANT_NAME = ModelVariant.RESNET_101
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def training_tester() -> ResNetTester:
    return ResNetTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


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
@pytest.mark.large
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "error: 'ttir.conv2d' op Padding attribute values must be >= 0."
    )
)
def test_resnet_v1_5_101_training(training_tester: ResNetTester):
    training_tester.test()
