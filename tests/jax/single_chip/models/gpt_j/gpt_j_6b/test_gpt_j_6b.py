# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import (
    BringupStatus,
    Category,
    failed_runtime,
)
from third_party.tt_forge_models.config import Parallelism

from third_party.tt_forge_models.gpt_j.causal_lm.jax import (
    ModelVariant,
    ModelLoader,
)
from ..tester import GPTJTester

VARIANT_NAME = ModelVariant._6B

MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> GPTJTester:
    return GPTJTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> GPTJTester:
    return GPTJTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.large
@pytest.mark.xfail(
    reason=failed_runtime(
        "Out of Memory: Not enough space to allocate 268435456 B DRAM buffer across 12 banks, "
        "where each bank needs to store 22372352 B "
        "(https://github.com/tenstorrent/tt-xla/issues/918)"
    )
)
def test_gpt_j_6b_inference(inference_tester: GPTJTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
)
@pytest.mark.large
@pytest.mark.skip(reason="Support for training not implemented")
def test_gpt_j_6b_training(training_tester: GPTJTester):
    training_tester.test()
