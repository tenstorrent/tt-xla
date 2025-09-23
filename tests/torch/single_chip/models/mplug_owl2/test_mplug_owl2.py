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

from third_party.tt_forge_models.mplug_owl2.loader import ModelVariant
from .tester import MplugOwl2Tester


VARIANT_NAME = ModelVariant.LLAMA2_7B


MODEL_NAME = build_model_name(
    Framework.TORCH,
    "mplug_owl2",
    "llama2_7b",
    ModelTask.MM_CAUSAL_LM,
    ModelSource.GITHUB,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MplugOwl2Tester:
    return MplugOwl2Tester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> MplugOwl2Tester:
    return MplugOwl2Tester(VARIANT_NAME, run_mode=RunMode.TRAINING)


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
def test_torch_mplug_owl2_inference(inference_tester: MplugOwl2Tester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_mplug_owl2_training(training_tester: MplugOwl2Tester):
    training_tester.test()


