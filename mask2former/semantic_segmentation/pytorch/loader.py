# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mask2Former model loader implementation for semantic segmentation tasks.
"""

import torch
from typing import Optional
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from datasets import load_dataset


class ModelVariant(StrEnum):
    """Available Mask2Former model variants for semantic segmentation."""

    SWIN_L_CITYSCAPES = "Swin_Large_Cityscapes"
    SWIN_L_MAPILLARY_VISTAS = "Swin_Large_Mapillary_Vistas"


class ModelLoader(ForgeModel):
    """Mask2Former model loader implementation for semantic segmentation tasks."""

    _VARIANTS = {
        ModelVariant.SWIN_L_CITYSCAPES: ModelConfig(
            pretrained_model_name="facebook/mask2former-swin-large-cityscapes-semantic"
        ),
        ModelVariant.SWIN_L_MAPILLARY_VISTAS: ModelConfig(
            pretrained_model_name="facebook/mask2former-swin-large-mapillary-vistas-semantic"
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SWIN_L_CITYSCAPES

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.image_processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Mask2Former",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_image_processor(self):
        self.image_processor = AutoImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.image_processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        model_kwargs |= kwargs

        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.image_processor is None:
            self._load_image_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.image_processor(images=image, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
