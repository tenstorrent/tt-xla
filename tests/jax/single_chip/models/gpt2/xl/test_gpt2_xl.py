# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category, ExecutionPass, failed_runtime

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.gpt2.causal_lm.jax import ModelLoader, ModelVariant

from ..tester import GPT2Tester

VARIANT_NAME = ModelVariant.XL
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----

@pytest.fixture
def training_tester() -> GPT2Tester:
    return GPT2Tester(VARIANT_NAME, run_mode=RunMode.TRAINING)

# ----- Tests -----

@pytest.mark.training
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
    execution_pass=ExecutionPass.FORWARD,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.large
@pytest.mark.xfail(
    reason=failed_runtime(
        "Out of Memory: Not enough space to allocate 160822400 B DRAM buffer "
        "across 12 banks, where each bank needs to store 13404800 B "
        "https://github.com/tenstorrent/tt-xla/issues/1650"
    )
)
def test_gpt2_xl_training(training_tester: GPT2Tester):
    training_tester.test()
