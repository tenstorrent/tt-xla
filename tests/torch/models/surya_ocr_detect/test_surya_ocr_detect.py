# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import inspect

import pytest
from infra import RunMode
from utils import BringupStatus, Category

import third_party.tt_forge_models.suryaocr.pytorch.loader as surya_loader
from tests.runner.requirements import RequirementsManager
from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.suryaocr.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)

from .tester import SuryaOCRDetectionTester

VARIANT = ModelVariant.OCR_DETECTION
MODEL_INFO = ModelLoader._get_model_info(VARIANT)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> SuryaOCRDetectionTester:
    return SuryaOCRDetectionTester(ModelVariant.OCR_DETECTION)


@pytest.fixture
def training_tester() -> SuryaOCRDetectionTester:
    return SuryaOCRDetectionTester(ModelVariant.OCR_DETECTION, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
def test_torch_surya_ocr_detect_inference():
    loader_path = inspect.getsourcefile(surya_loader)
    with RequirementsManager.for_loader(loader_path):
        tester = SuryaOCRDetectionTester(ModelVariant.OCR_DETECTION)
        tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_surya_ocr_detect_training():
    loader_path = inspect.getsourcefile(surya_loader)
    with RequirementsManager.for_loader(loader_path):
        tester = SuryaOCRDetectionTester(ModelVariant.OCR_DETECTION, run_mode=RunMode.TRAINING)
        tester.test()
