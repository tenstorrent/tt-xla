# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from tests.infra.testers.compiler_config import CompilerConfig
from utils import (
    BringupStatus,
    Category,
    failed_runtime,
    incorrect_result,
)
from third_party.tt_forge_models.config import Parallelism

from ..tester import ResNetTester
from third_party.tt_forge_models.resnet.image_classification.jax import (
    ModelVariant,
    ModelLoader,
)

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
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
@pytest.mark.large
@pytest.mark.xfail(
    reason=incorrect_result(
        "Calculated: pcc=-0.05205399543046951. Required: pcc=0.99. "
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
def test_resnet_v1_5_26_inference(inference_tester: ResNetTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_resnet_v1_5_26_training(training_tester: ResNetTester):
    training_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
)
@pytest.mark.xfail(
    reason=failed_runtime(
        "Float32 not supported for ttnn.maxpool_2d op "
        "https://github.com/tenstorrent/tt-xla/issues/941"
    )
)
def test_resnet_v1_5_26_inference_optimizer(inference_tester_optimizer: ResNetTester):
    inference_tester_optimizer.test()
