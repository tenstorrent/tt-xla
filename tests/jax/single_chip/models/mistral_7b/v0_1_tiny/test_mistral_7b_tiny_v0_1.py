# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TODO: Refactor to use ModelLoader.get_model_info() once the PR in tt-forge-models is merged

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
from third_party.tt_forge_models.config import Parallelism

from ..tester import Mistral7BTester

MODEL_PATH = "ksmcg/Mistral-tiny"
MODEL_GROUP = ModelGroup.GENERALITY
MODEL_NAME = build_model_name(
    Framework.JAX,
    "mistral-7b",
    "v0.1_tiny",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> Mistral7BTester:
    return Mistral7BTester(MODEL_PATH)


def training_tester() -> Mistral7BTester:
    return Mistral7BTester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=MODEL_GROUP,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.xfail(
    reason=failed_runtime(
        "Out of Memory: Not enough space to allocate 2147483648 B DRAM buffer "
        "across 12 banks, where each bank needs to store 178958336 B "
        "(https://github.com/tenstorrent/tt-xla/issues/917)"
    )
)
def test_mistral_7b_v0_1_tiny_inference(inference_tester: Mistral7BTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=MODEL_GROUP,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_mistral_7b_v0_1_tiny_training(training_tester: Mistral7BTester):
    training_tester.test()
