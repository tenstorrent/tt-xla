# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from surya.settings import Settings

from .model_utils import TORCH_DEVICE_MODEL, _patched_init, _prepare_image

Settings.TORCH_DEVICE_MODEL = TORCH_DEVICE_MODEL
import pytest
from infra import RunMode
from surya.detection import DetectionPredictor
from surya.foundation.cache import ContinuousBatchingCache as _CBC
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.suryaocr.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)

from .tester import SuryaOCRTester

_CBC.__init__ = _patched_init
DetectionPredictor.prepare_image = _prepare_image
VARIANT = ModelVariant.OCR_TEXT
MODEL_INFO = ModelLoader._get_model_info(VARIANT)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> SuryaOCRTester:
    return SuryaOCRTester(ModelVariant.OCR_TEXT)


@pytest.fixture
def training_tester() -> SuryaOCRTester:
    return SuryaOCRTester(ModelVariant.OCR_TEXT, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.UNKNOWN,
)
@pytest.mark.skip(reason="Model gets killed and hence skipped for now")
def test_torch_surya_ocr_text_inference(inference_tester: SuryaOCRTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_surya_ocr_text_training(training_tester: SuryaOCRTester):
    training_tester.test()
