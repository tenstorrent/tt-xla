# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from utils import (
    BringupStatus,
    Category,
    ExecutionPass,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_runtime,
)

from third_party.tt_forge_models.mistral.causal_lm.jax import ModelVariant

from ..tester import Mistral7BTester

VARIANT_NAME = ModelVariant.V0_3_INSTRUCT
MODEL_GROUP = ModelGroup.RED
MODEL_NAME = build_model_name(
    Framework.JAX,
    "mistral-7b",
    "v0.3_instruct",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> Mistral7BTester:
    return Mistral7BTester(VARIANT_NAME)


def training_tester() -> Mistral7BTester:
    return Mistral7BTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=MODEL_GROUP,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.large
# @pytest.mark.skip(
#     reason=failed_runtime(
#         "Not enough space to allocate 234881024 B DRAM buffer across 12 banks, "
#         "where each bank needs to store 19574784 B "
#         "(https://github.com/tenstorrent/tt-xla/issues/917)"
#     )
# )
def test_mistral_7b_v0_3_instruct_inference(inference_tester: Mistral7BTester):
    inference_tester.test()


@pytest.mark.training
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=MODEL_GROUP,
    run_mode=RunMode.TRAINING,
    execution_pass=ExecutionPass.FORWARD,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.large
@pytest.mark.skip(
    reason=failed_runtime(
        "Not enough space to allocate 117440512 B DRAM buffer across 12 banks, "
        "where each bank needs to store 9805824 B "
        "(https://github.com/tenstorrent/tt-xla/issues/917)"
    )
)
def test_mistral_7b_v0_3_instruct_training(training_tester: Mistral7BTester):
    training_tester.test()
