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
    failed_ttmlir_compilation,
)

from third_party.tt_forge_models.flux.pytorch import ModelVariant
from ..tester import FluxTester


MODEL_NAME = build_model_name(
    Framework.TORCH,
    "flux",
    "schnell",
    ModelTask.MM_IMAGE_TTT,  # FIXME: Update task to Text to Image
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FluxTester:
    return FluxTester(ModelVariant.SCHNELL)


@pytest.fixture
def training_tester() -> FluxTester:
    return FluxTester(ModelVariant.SCHNELL, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "error: failed to legalize operation 'stablehlo.batch_norm_training' "
        "https://github.com/tenstorrent/tt-xla/issues/735"
    )
)
def test_torch_flux_schnell_inference(inference_tester: FluxTester):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.RED,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_flux_schnell_training(training_tester: FluxTester):
    training_tester.test()
