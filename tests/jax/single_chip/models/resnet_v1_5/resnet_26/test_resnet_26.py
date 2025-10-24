# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category, failed_runtime

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.resnet.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)

from ..tester import ResNetTester

VARIANT_NAME = ModelVariant.RESNET_26
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> ResNetTester:
    return ResNetTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> ResNetTester:
    return ResNetTester(VARIANT_NAME, RunMode.TRAINING)


@pytest.fixture
def inference_tester_optimizer() -> ResNetTester:
    return ResNetTester(
        VARIANT_NAME,
        run_mode=RunMode.INFERENCE,
        compiler_config=CompilerConfig(enable_optimizer=True),
    )


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
@pytest.mark.large
def test_resnet_v1_5_26_inference(inference_tester: ResNetTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_resnet_v1_5_26_training(training_tester: ResNetTester):
    training_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
)
@pytest.mark.skip(
    reason=failed_runtime(
        "Optimizer test hangs as part of a test suite - works when run standalone "
        "https://github.com/tenstorrent/tt-xla/issues/1547",
    )
)
def test_resnet_v1_5_26_inference_optimizer(inference_tester_optimizer: ResNetTester):
    inference_tester_optimizer.test()
