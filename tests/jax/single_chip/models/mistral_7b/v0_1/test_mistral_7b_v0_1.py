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
    failed_runtime,
)

from ..tester import Mistral7BTester

MODEL_PATH = "ksmcg/Mistral-7B-v0.1"
MODEL_GROUP = ModelGroup.GENERALITY
MODEL_NAME = build_model_name(
    Framework.JAX,
    "mistral-7b",
    "v0.1",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> Mistral7BTester:
    return Mistral7BTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> Mistral7BTester:
    return Mistral7BTester(MODEL_PATH, run_mode=RunMode.TRAINING)


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
@pytest.mark.skip(
    reason=failed_runtime(
        "Not enough space to allocate 117440512 B DRAM buffer across 12 banks, "
        "where each bank needs to store 9805824 B "
        "(https://github.com/tenstorrent/tt-xla/issues/917)"
    )
)
def test_mistral_7b_v0_1_inference(inference_tester: Mistral7BTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=MODEL_GROUP,
    run_mode=RunMode.TRAINING,
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
def test_mistral_7b_v0_1_training(training_tester: Mistral7BTester):
    training_tester.test()
