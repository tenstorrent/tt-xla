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
    failed_fe_compilation,
)

from ..tester import GPTJTester

MODEL_PATH = "EleutherAI/gpt-j-6B"
MODEL_NAME = build_model_name(
    Framework.JAX, "gpt-j", "6b", ModelTask.NLP_CAUSAL_LM, ModelSource.HUGGING_FACE
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> GPTJTester:
    return GPTJTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> GPTJTester:
    return GPTJTester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_FE_COMPILATION,
)
# @pytest.mark.xfail(
#     reason=failed_ttmlir_compilation(
#         "failed to legalize operation 'ttir.gather' that was explicitly marked illegal "
#         "https://github.com/tenstorrent/tt-xla/issues/318 "
#     )
# )
@pytest.mark.skip(
    reason=failed_fe_compilation(
        "OOMs in CI (https://github.com/tenstorrent/tt-xla/issues/186)"
    )
)
def test_gpt_j_6b_inference(inference_tester: GPTJTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_gpt_j_6b_training(training_tester: GPTJTester):
    training_tester.test()
