# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import (
    BringupStatus,
    Category,
    failed_ttmlir_compilation,
)
from third_party.tt_forge_models.wav2vec2.audio_classification.jax import (
    ModelVariant,
    ModelLoader,
)
from third_party.tt_forge_models.config import Parallelism
from ..tester import Wav2Vec2Tester

VARIANT_NAME = ModelVariant.LARGE_LV_60
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> Wav2Vec2Tester:
    return Wav2Vec2Tester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> Wav2Vec2Tester:
    return Wav2Vec2Tester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "Failed to legalize operation 'ttir.scatter' "
        "https://github.com/tenstorrent/tt-xla/issues/911"
    )
)
def test_wav2vec2_large_lv60_inference(inference_tester: Wav2Vec2Tester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_wav2vec2_large_lv60_training(training_tester: Wav2Vec2Tester):
    training_tester.test()
