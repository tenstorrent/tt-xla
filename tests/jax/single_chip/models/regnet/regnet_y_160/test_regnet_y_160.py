# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category, failed_runtime

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.regnet.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)

from ..tester import RegNetTester

VARIANT_NAME = ModelVariant.REGNET_Y_160
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> RegNetTester:
    return RegNetTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> RegNetTester:
    return RegNetTester(VARIANT_NAME, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.xfail(
    reason=failed_runtime(
        "Out of Memory: Not enough space to allocate 15259926528 B B L1 buffer "
        "across 1 banks, where each bank needs to store 1271660544 B  "
        "(https://github.com/tenstorrent/tt-xla/issues/187)"
    )
)
def test_regnet_y_160_inference(inference_tester: RegNetTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_regnet_y_160_training(training_tester: RegNetTester):
    training_tester.test()
