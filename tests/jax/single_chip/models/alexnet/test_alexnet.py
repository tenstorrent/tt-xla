# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TODO: Refactor to use ModelLoader.get_model_info() once the PR in tt-forge-models is merged

import pytest
from infra import RunMode
from utils import BringupStatus, Category, ExecutionPass, failed_ttmlir_compilation

from third_party.tt_forge_models.alexnet.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)
from third_party.tt_forge_models.config import Parallelism

from .tester import AlexNetTester

VARIANT = ModelVariant.CUSTOM

VARIANT_NAME = ModelVariant.CUSTOM
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def training_tester() -> AlexNetTester:
    return AlexNetTester(VARIANT, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.test_forge_models_training
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
    execution_pass=ExecutionPass.FORWARD,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.xfail(
    reason=failed_runtime(
        "Statically allocated circular buffers on core range [(x=0,y=0) - (x=1,y=1)] "
        "grow to 3087776 B which is beyond max L1 size of 1499136 B"
    )
)
def test_alexnet_training(training_tester: AlexNetTester):
    training_tester.test()
