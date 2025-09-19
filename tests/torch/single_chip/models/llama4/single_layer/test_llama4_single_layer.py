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

from .tester import Llama4SingleLayerTester

VARIANT_NAME = "scout-l4-7b"


MODEL_NAME = build_model_name(
    Framework.TORCH,
    "llama4-single-layer",
    "model",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> Llama4SingleLayerTester:
    return Llama4SingleLayerTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> Llama4SingleLayerTester:
    return Llama4SingleLayerTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.skip(reason="Fails to run on CPU")
def test_torch_llama4_single_layer_inference(inference_tester: Llama4SingleLayerTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_llama4_single_layer_training(training_tester: Llama4SingleLayerTester):
    training_tester.test()
