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
from third_party.tt_forge_models.config import Parallelism

from third_party.tt_forge_models.marian_mt.text_classification.jax import (
    ModelVariant,
    ModelLoader,
)
from ..tester import MarianTester

VARIANT_NAME = ModelVariant.OPUS_MT_EN_DE

MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MarianTester:
    return MarianTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> MarianTester:
    return MarianTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


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
def test_marian_opus_mt_en_de_inference(inference_tester: MarianTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_marian_opus_mt_en_de_training(training_tester: MarianTester):
    training_tester.test()
