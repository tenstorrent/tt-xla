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
    incorrect_result,
)
from third_party.tt_forge_models.clip.image_classification.jax import ModelVariant

from ..tester import FlaxCLIPTester

VARIANT_NAME = ModelVariant.LARGE_PATCH14_336
MODEL_NAME = build_model_name(
    Framework.JAX,
    "clip_vit_patch14_336",
    "large",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxCLIPTester:
    return FlaxCLIPTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> FlaxCLIPTester:
    return FlaxCLIPTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
@pytest.mark.xfail(
    reason=incorrect_result(
        "PCC comparison failed. Calculated: pcc=0.9046211838722229. Required: pcc=0.99 "
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
def test_clip_large_patch14_336_inference(inference_tester: FlaxCLIPTester):
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
def test_clip_large_patch14_336_training(training_tester: FlaxCLIPTester):
    training_tester.test()
