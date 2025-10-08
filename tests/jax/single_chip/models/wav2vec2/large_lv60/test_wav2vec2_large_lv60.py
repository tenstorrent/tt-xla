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

from ..tester import Wav2Vec2Tester

MODEL_PATH = "facebook/wav2vec2-large-lv60"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "wav2vec2",
    "large-lv60",
    ModelTask.AUDIO_CLS,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> Wav2Vec2Tester:
    return Wav2Vec2Tester(MODEL_PATH)


@pytest.fixture
def training_tester() -> Wav2Vec2Tester:
    return Wav2Vec2Tester(MODEL_PATH, run_mode=RunMode.TRAINING)


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
        "failed to legalize operation 'stablehlo.dynamic_slice' "
        "https://github.com/tenstorrent/tt-xla/issues/404"
    )
)
def test_wav2vec2_large_lv60_inference(inference_tester: Wav2Vec2Tester):
    inference_tester.test()


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_wav2vec2_large_lv60_training(training_tester: Wav2Vec2Tester):
    training_tester.test()
