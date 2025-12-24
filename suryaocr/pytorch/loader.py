# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SuryaOCR model loader implementation
"""

from typing import Optional, List

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ...base import ForgeModel
from ...tools.utils import get_file


from .src.utils import (
    SuryaOCRWrapper,
    save_outputs_ocr_text,
    save_outputs_ocr_detection,
    SuryaOCRDetectionWrapper,
)
from pathlib import Path


class ModelVariant(StrEnum):
    """Available SuryaOCR model variants.

    Currently a single default variant that wraps Surya's detection and recognition predictors.
    """

    OCR_TEXT = "ocr_text"
    OCR_DETECTION = "ocr_detection"


class ModelLoader(ForgeModel):
    """SuryaOCR model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.OCR_TEXT: ModelConfig(
            pretrained_model_name="surya_ocr_text",
        ),
        ModelVariant.OCR_DETECTION: ModelConfig(
            pretrained_model_name="surya_ocr_detection",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.OCR_TEXT

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="suryaocr",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self._transform = transforms.Compose([transforms.ToTensor()])
        self.image_tensor = None

    def load_model(self, dtype_override=None) -> nn.Module:
        """Load Surya OCR wrapper model.

        Returns:
            nn.Module: A wrapper module that calls Surya predictors.
        """

        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor

        if DetectionPredictor is None or RecognitionPredictor is None:
            raise ImportError(
                "Surya package is not available. Please install `surya` to use SuryaOCR loader."
            )
        if self.image_tensor is None:
            self.load_inputs()
        if self._variant == ModelVariant.OCR_TEXT:
            model = SuryaOCRWrapper(image_tensor=self.image_tensor)
        elif self._variant == ModelVariant.OCR_DETECTION:
            model = SuryaOCRDetectionWrapper()
        else:
            raise ValueError(f"Invalid variant: {self._variant}")
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override: Optional[torch.dtype] = torch.float32):
        """Generate sample inputs for Surya OCR.

        Returns:
            List[torch.Tensor, torch.Tensor]: [images_tensor, languages_tensor]
        """
        image_file = get_file(
            "https://raw.githubusercontent.com/VikParuchuri/surya/master/static/images/excerpt_text.png"
        )
        image = Image.open(str(image_file)).convert("RGB")
        image_tensor = self._transform(image)
        image_tensor = torch.stack([image_tensor])

        images: List[Image.Image] = [image]
        self.images = images
        if dtype_override is not None:
            image_tensor = image_tensor.to(dtype_override)
        self.image_tensor = image_tensor
        return image_tensor

    def post_process(self, co_out, result_path):
        if self._variant == ModelVariant.OCR_TEXT:
            save_outputs_ocr_text(co_out, self.images, result_path)
        elif self._variant == ModelVariant.OCR_DETECTION:
            save_outputs_ocr_detection(co_out, self.images, result_path)
        else:
            raise ValueError(f"Invalid variant: {self.variant}")
