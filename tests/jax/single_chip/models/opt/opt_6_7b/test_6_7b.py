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

from third_party.tt_forge_models.opt.causal_lm.jax import (
    ModelVariant,
    ModelLoader,
)
from ..tester import OPTTester

MODEL_PATH = "facebook/opt-6.7b"
VARIANT_NAME = ModelVariant._6_7B
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> OPTTester:
    return OPTTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> OPTTester:
    return OPTTester(MODEL_PATH, run_mode=RunMode.TRAINING)


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
        "Not enough space to allocate 134217728 B DRAM buffer across 12 banks, "
        "where each bank needs to store 11186176 B "
        "(https://github.com/tenstorrent/tt-xla/issues/918)"
    )
)
def test_opt_6_7b_inference(inference_tester: OPTTester):
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
def test_opt_6_7b_training(training_tester: OPTTester):
    training_tester.test()
