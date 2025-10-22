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

VARIANT_NAME = ModelVariant.V0_1_TINY
MODEL_GROUP = ModelGroup.GENERALITY
MODEL_NAME = build_model_name(
    Framework.JAX,
    "mistral-7b",
    "v0.1_tiny",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----

# ----- Tests -----


@pytest.mark.xfail(
    reason=failed_runtime(
        "Out of Memory: Not enough space to allocate 2147483648 B DRAM buffer "
        "across 12 banks, where each bank needs to store 178958336 B "
        "(https://github.com/tenstorrent/tt-xla/issues/917)"
    )
)
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
def test_mistral_7b_v0_1_tiny_training(training_tester: Mistral7BTester):
    training_tester.test()
