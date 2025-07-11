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
    failed_ttmlir_compilation,
)

from .tester import MGPSTRTester

VARIANT_NAME = "alibaba-damo/mgp-str-base"


MODEL_NAME = build_model_name(
    Framework.TORCH,
    "mgp_str",
    "base",
    ModelTask.CV_IMAGE_FE,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MGPSTRTester:
    return MGPSTRTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> MGPSTRTester:
    return MGPSTRTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


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
        "error: failed to legalize operation 'stablehlo.batch_norm_training' "
        "https://github.com/tenstorrent/tt-xla/issues/735"
    )
)
def test_torch_mgp_str_inference(inference_tester: MGPSTRTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_mgp_str_training(training_tester: MGPSTRTester):
    training_tester.test()
