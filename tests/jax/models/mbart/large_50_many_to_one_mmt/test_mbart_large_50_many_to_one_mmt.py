# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from infra import Framework, RunMode

from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_ttmlir_compilation,
)

from ..tester import MBartTester

MODEL_PATH = "facebook/mbart-large-50-many-to-one-mmt"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "mbart",
    "large_50_many_to_one_mmt",
    ModelTask.NLP_SUMMARIZATION,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MBartTester:
    return MBartTester(MODEL_PATH)


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
        "error: 'ttir.scatter' op Dimension size to slice into must be 1 "
        "https://github.com/tenstorrent/tt-xla/issues/386"
    )
)
def test_mbart_large_50_many_to_one_mmt_inference(inference_tester: MBartTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_mbart_large_50_many_to_one_mmt_training(inference_tester: MBartTester):
    training_tester.test()
