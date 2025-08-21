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

from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor


class ModelVariant(StrEnum):
    """Available SuryaOCR model variants.

    Currently a single default variant that wraps Surya's detection and recognition predictors.
    """

    DEFAULT = "default"


class ModelLoader(ForgeModel):
    """SuryaOCR model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="surya_ocr_default",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DEFAULT

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

    def load_model(self, dtype_override=None) -> nn.Module:
        """Load Surya OCR wrapper model.

        Returns:
            nn.Module: A wrapper module that calls Surya predictors.
        """

        if DetectionPredictor is None or RecognitionPredictor is None:
            raise ImportError(
                "Surya package is not available. Please install `surya` to use SuryaOCR loader."
            )

        class SuryaOCRWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.recognition_predictor = RecognitionPredictor()
                self.detection_predictor = DetectionPredictor()

            def forward(
                self, images_tensor: torch.Tensor, languages_tensor: torch.Tensor
            ):
                batch_size = images_tensor.shape[0]
                images_list: List[Image.Image] = [
                    transforms.ToPILImage()(images_tensor[i]) for i in range(batch_size)
                ]
                language_indices = languages_tensor.tolist()
                lang_keys = list({"en": 0, "fr": 1, "de": 2}.keys())
                languages_list = [
                    [lang_keys[i] for i in batch] for batch in language_indices
                ]

                ocr_results = self.recognition_predictor(
                    images_list, languages_list, self.detection_predictor
                )
                return ocr_results

        model = SuryaOCRWrapper()
        model.eval()

        if hasattr(model, "recognition_predictor") and hasattr(
            model.recognition_predictor, "model"
        ):
            for _, param in model.recognition_predictor.model.named_parameters():
                param.requires_grad = False
        if hasattr(model, "detection_predictor") and hasattr(
            model.detection_predictor, "model"
        ):
            for _, param in model.detection_predictor.model.named_parameters():
                param.requires_grad = False

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for Surya OCR.

        Returns:
            List[torch.Tensor, torch.Tensor]: [images_tensor, languages_tensor]
        """
        image_file = get_file(
            "https://raw.githubusercontent.com/VikParuchuri/surya/master/static/images/excerpt_text.png"
        )
        image = Image.open(str(image_file)).convert("RGB")
        image_tensor = self._transform(image)
        images = torch.stack([image_tensor])

        language_tensor = torch.tensor([[0]], dtype=torch.int64)

        if dtype_override is not None:
            images = images.to(dtype_override)

        return [images, language_tensor]
