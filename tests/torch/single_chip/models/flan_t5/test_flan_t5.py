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
)

from third_party.tt_forge_models.t5.pytorch import ModelVariant

from .tester import FlanT5Tester

VARIANT_NAME = ModelVariant.FLAN_T5_SMALL


MODEL_NAME = build_model_name(
    Framework.TORCH,
    "flan_t5",
    "small",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlanT5Tester:
    return FlanT5Tester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> FlanT5Tester:
    return FlanT5Tester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_torch_flan_t5_inference(inference_tester: FlanT5Tester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_flan_t5_training(training_tester: FlanT5Tester):
    training_tester.test()
