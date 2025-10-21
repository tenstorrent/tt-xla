# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category, failed_fe_compilation

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.whisper.audio_classification.jax import (
    ModelLoader,
    ModelVariant,
)

from ..tester import WhisperTester

VARIANT_NAME = ModelVariant.MEDIUM
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----

@pytest.fixture
def training_tester() -> WhisperTester:
    return WhisperTester(VARIANT_NAME, run_mode=RunMode.TRAINING)

# ----- Tests -----

@pytest.mark.xfail(
    reason=failed_fe_compilation(
        "NotImplementedError: Could not run 'torchcodec_ns::create_from_tensor'"
        "https://github.com/tenstorrent/tt-xla/issues/1635"
    )
)

@pytest.mark.training
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_whisper_medium_training(training_tester: WhisperTester):
    training_tester.test()
