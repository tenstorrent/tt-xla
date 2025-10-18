# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category, ExecutionPass, failed_runtime

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.dinov2.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)

from ..tester import Dinov2Tester

VARIANT_NAME = ModelVariant.GIANT
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> Dinov2Tester:
    return Dinov2Tester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> Dinov2Tester:
    return Dinov2Tester(VARIANT_NAME, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.large
@pytest.mark.xfail(
    reason=failed_runtime(
        "Statically allocated circular buffers in program 1331 clash with "
        "L1 buffers on core range[(x=0,y=0) - (x=7,y=0)]. "
        "L1 buffer allocated at 1069056 and static circular buffer region ends at 1117312"
        "(https://github.com/tenstorrent/tt-xla/issues/1066)"
    )
)
def test_dinov2_giant_inference(inference_tester: Dinov2Tester):
    inference_tester.test()


@pytest.mark.training
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.TRAINING,
    execution_pass=ExecutionPass.FORWARD,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.large
@pytest.mark.skip(
    reason=failed_runtime(
        "Statically allocated circular buffers in program 2669 clash with "
        "L1 buffers on core range[(x=0,y=0) - (x=7,y=0)]. "
        "L1 buffer allocated at 1069056 and static circular buffer region ends at 1117312"
        "(https://github.com/tenstorrent/tt-xla/issues/1066)"
    )
)
def test_dinov2_giant_training(training_tester: Dinov2Tester):
    training_tester.test()
