# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    incorrect_result,
)

from third_party.tt_forge_models.autoencoder.pytorch.loader import ModelVariant

from .tester import AutoencoderLinearTester

VARIANT_NAME = ModelVariant.LINEAR

MODEL_NAME = build_model_name(
    Framework.TORCH,
    "autoencoder",
    str(VARIANT_NAME),
    ModelTask.CV_IMG_TO_IMG,
    ModelSource.CUSTOM,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> AutoencoderLinearTester:
    return AutoencoderLinearTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> AutoencoderLinearTester:
    return AutoencoderLinearTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
@pytest.mark.xfail(
    reason=incorrect_result(
        "PCC comparison failed on Blackhole. Calculated: pcc=0.039223916828632355. Required: pcc=0.99."
        "https://github.com/tenstorrent/tt-xla/issues/1038"
    )
)
def test_torch_autoencoder_linear_inference(inference_tester: AutoencoderLinearTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_autoencoder_linear_training(training_tester: AutoencoderLinearTester):
    training_tester.test()
