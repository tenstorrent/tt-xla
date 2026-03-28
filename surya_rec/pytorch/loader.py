# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Surya Recognition model loader implementation for OCR text recognition tasks.
"""
import torch
from PIL import Image
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Surya Recognition model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Surya Recognition model loader for OCR text recognition."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="vikp/surya_rec",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="surya_rec",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        foundation_predictor = FoundationPredictor(device="cpu")
        rec_predictor = RecognitionPredictor(foundation_predictor)
        model = rec_predictor.model
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        image = Image.new("RGB", (896, 196), color=(255, 255, 255))

        # Normalize to [-1, 1] matching surya's image processor (mean=0.5, std=0.5)
        pixel_values = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(
            1, 196, 896, 3
        )
        pixel_values = pixel_values.permute(0, 3, 1, 2) / 255.0
        pixel_values = (pixel_values - 0.5) / 0.5

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        if batch_size > 1:
            pixel_values = pixel_values.repeat(batch_size, 1, 1, 1)

        return pixel_values
