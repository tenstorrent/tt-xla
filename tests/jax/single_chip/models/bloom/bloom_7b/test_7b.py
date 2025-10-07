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

from ..tester import BloomTester
from third_party.tt_forge_models.bloom.causal_lm.jax import ModelVariant

MODEL_VARIANT = ModelVariant.BLOOM_7B
MODEL_NAME = build_model_name(
    Framework.JAX,
    "bloom",
    "7b",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> BloomTester:
    return BloomTester(MODEL_VARIANT)


@pytest.fixture
def training_tester() -> BloomTester:
    return BloomTester(MODEL_VARIANT, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.large
@pytest.mark.xfail(
    reason=failed_runtime(
        "Out of Memory: Not enough space to allocate 2055208960 B DRAM buffer across 12 banks, "
        "where each bank needs to store 171270144 B "
        "(https://github.com/tenstorrent/tt-xla/issues/918)"
    )
)
def test_bloom_7b_inference(inference_tester: BloomTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
    execution_pass=ExecutionPass.FORWARD,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.large
@pytest.mark.xfail(
    reason=failed_runtime(
        "Out of Memory: Not enough space to allocate 2055208960 B DRAM buffer across 12 banks, "
        "where each bank needs to store 171270144 B "
        "https://github.com/tenstorrent/tt-xla/issues/918"
    )
)
def test_bloom_7b_training(training_tester: BloomTester):
    training_tester.test()
