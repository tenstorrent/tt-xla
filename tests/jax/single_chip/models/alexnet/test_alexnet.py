# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TODO: Refactor to use ModelLoader.get_model_info() once the PR in tt-forge-models is merged

import pytest
from infra import RunMode
from utils import BringupStatus, Category, failed_ttmlir_compilation

from third_party.tt_forge_models.alexnet.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)
from third_party.tt_forge_models.config import Parallelism

from .tester import AlexNetTester

VARIANT_NAME = ModelVariant.CUSTOM
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> AlexNetTester:
    return AlexNetTester()


@pytest.fixture
def training_tester() -> AlexNetTester:
    return AlexNetTester(run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "failed to legalize operation 'ttir.gather' "
        "https://github.com/tenstorrent/tt-xla/issues/318"
    )
)
def test_alexnet_inference(inference_tester: AlexNetTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_alexnet_training(training_tester: AlexNetTester):
    training_tester.test()
