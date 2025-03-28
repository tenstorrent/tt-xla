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
    failed_fe_compilation,
)

from ..tester import ViTHybridTester

MODEL_PATH = "google/vit-hybrid-base-bit-384"
MODEL_NAME = build_model_name(
    Framework.TORCH,
    "vit_hybrid",
    "base_bit_384",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> ViTHybridTester:
    return ViTHybridTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> ViTHybridTester:
    return ViTHybridTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_FE_COMPILATION,
)
@pytest.mark.skip(
    reason=failed_fe_compilation(
        "OOMs in CI (https://github.com/tenstorrent/tt-xla/issues/186)"
    )
)
def test_vit_hybrid_base_bit_384_inference(
    inference_tester: ViTHybridTester,
):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_vit_hybrid_base_bit_384_training(training_tester: ViTHybridTester):
    training_tester.test()
