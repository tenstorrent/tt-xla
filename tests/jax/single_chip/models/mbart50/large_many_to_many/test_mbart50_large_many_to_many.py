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
    failed_ttmlir_compilation,
)

from ..tester import MBartTester

MODEL_PATH = "facebook/mbart-large-50-many-to-many-mmt"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "mbart50",
    "large_many_to_many",
    ModelTask.NLP_SUMMARIZATION,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MBartTester:
    return MBartTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> MBartTester:
    return MBartTester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "'ttir.scatter' op Dimension size to slice into must be 1 "
        "https://github.com/tenstorrent/tt-xla/issues/386"
    )
)
def test_mbart50_large_many_to_many_inference(inference_tester: MBartTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
    execution_pass=ExecutionPass.BACKWARD,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "error: failed to legalize operation 'ttir.scatter'"
        "https://github.com/tenstorrent/tt-mlir/issues/4792"
    )
)
def test_mbart50_large_many_to_many_training(training_tester: MBartTester):
    training_tester.test()
