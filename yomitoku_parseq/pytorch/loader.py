# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Yomitoku PaRSEQ (Permutation-invariant Autoregressive Sequence-to-sequence)
model loader for Japanese scene/document text recognition.
"""
import torch
from PIL import Image
from typing import Optional
from torchvision import transforms

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
    """Available Yomitoku PaRSEQ model variants."""

    OPEN_BETA = "Open_Beta"


class ModelLoader(ForgeModel):
    """Yomitoku PaRSEQ model loader for Japanese text recognition."""

    _VARIANTS = {
        ModelVariant.OPEN_BETA: ModelConfig(
            pretrained_model_name="KotaroKinoshita/yomitoku-text-recognizer-parseq-open-beta",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPEN_BETA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="yomitoku_parseq",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from yomitoku.models.parseq import PARSeq

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = PARSeq.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Create a sample text image for recognition
        image = Image.new("RGB", (320, 64), color=(255, 255, 255))
        pixel_values = transform(image).unsqueeze(0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
