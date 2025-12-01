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

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.stable_diffusion_xl.pytorch import (
    ModelLoader,
    ModelVariant,
)

from .tester import StableDiffusionXLTester

VARIANT_NAME = ModelVariant.STABLE_DIFFUSION_XL_BASE_1_0
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> StableDiffusionXLTester:
    return StableDiffusionXLTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> StableDiffusionXLTester:
    return StableDiffusionXLTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


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
@pytest.mark.xfail(
    reason="Out of Memory: Not enough space to allocate 94633984 B DRAM buffer across 12 banks"
)
def test_torch_stable_diffusion_xl_inference(inference_tester: StableDiffusionXLTester):
    inference_tester.test()


@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_stable_diffusion_xl_training(training_tester: StableDiffusionXLTester):
    training_tester.test()
