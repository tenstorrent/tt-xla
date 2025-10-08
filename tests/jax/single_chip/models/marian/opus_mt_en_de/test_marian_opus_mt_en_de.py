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

from third_party.tt_forge_models.marian_mt.text_classification.jax import ModelVariant

from ..tester import MarianTester

MODEL_VARIANT = ModelVariant.OPUS_MT_EN_DE
MODEL_NAME = build_model_name(
    Framework.JAX,
    "marian",
    "opus_mt_en_de",
    ModelTask.NLP_TEXT_CLS,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MarianTester:
    return MarianTester(MODEL_VARIANT)


@pytest.fixture
def training_tester() -> MarianTester:
    return MarianTester(MODEL_VARIANT, run_mode=RunMode.TRAINING)


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
        "Failed to legalize operation 'ttir.scatter' "
        "https://github.com/tenstorrent/tt-xla/issues/911"
    )
)
def test_marian_opus_mt_en_de_inference(inference_tester: MarianTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_marian_opus_mt_en_de_training(training_tester: MarianTester):
    training_tester.test()
