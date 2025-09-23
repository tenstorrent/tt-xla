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

from third_party.tt_forge_models.maptr.pytorch.loader import ModelVariant
from .tester import MAPTRTester


VARIANT_NAME = ModelVariant.TINY_R50_24E_BEVPOOL


MODEL_NAME = build_model_name(
    Framework.TORCH,
    "maptr",
    "bevpool",
    ModelTask.REALTIME_MAP_CONSTRUCTION,
    ModelSource.GITHUB,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MAPTRTester:
    return MAPTRTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> MAPTRTester:
    return MAPTRTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
# @pytest.mark.xfail(
#     reason=failed_ttmlir_compilation(
#         "error: failed to legalize operation 'stablehlo.batch_norm_training' "
#         "https://github.com/tenstorrent/tt-xla/issues/735"
#     )
# )
def test_torch_maptr_inference(inference_tester: MAPTRTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_maptr_training(training_tester: MAPTRTester):
    training_tester.test()


