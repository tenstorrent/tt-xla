# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.stable_diffusion.pytorch import (
    ModelLoader,
    ModelVariant,
)

from .tester import StableDiffusion35Tester

VARIANT_NAME = ModelVariant.STABLE_DIFFUSION_3_MEDIUM
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> StableDiffusion35Tester:
    return StableDiffusion35Tester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> StableDiffusion35Tester:
    return StableDiffusion35Tester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.single_device
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.skip(
    reason="RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13 - DRAM buffer allocation failure in ttnn::eq during MMDiT forward (same family as SD 3.5 Medium OOM)"
)
def test_torch_stable_diffusion_3_medium_inference(
    inference_tester: StableDiffusion35Tester,
):
    inference_tester.test()


@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_stable_diffusion_3_medium_training(
    training_tester: StableDiffusion35Tester,
):
    training_tester.test()
