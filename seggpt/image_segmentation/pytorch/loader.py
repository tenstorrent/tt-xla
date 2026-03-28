# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SegGPT model loader implementation for image segmentation tasks.
"""

import torch
from typing import Optional
from transformers import SegGptImageProcessor, SegGptForImageSegmentation
from PIL import Image

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
    """Available SegGPT model variants for image segmentation."""

    VIT_LARGE = "ViT_Large"


class ModelLoader(ForgeModel):
    """SegGPT model loader implementation for image segmentation tasks."""

    _VARIANTS = {
        ModelVariant.VIT_LARGE: ModelConfig(
            pretrained_model_name="BAAI/seggpt-vit-large"
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIT_LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.image_processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SegGPT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_image_processor(self):
        self.image_processor = SegGptImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.image_processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        model_kwargs |= kwargs

        model = SegGptForImageSegmentation.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.image_processor is None:
            self._load_image_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        # SegGPT requires prompt_images and prompt_masks for one-shot segmentation.
        # Use the same image as both input and prompt, with a simple mask.
        prompt_image = image
        prompt_mask = Image.new("L", image.size, 255)

        inputs = self.image_processor(
            images=image,
            prompt_images=prompt_image,
            prompt_masks=prompt_mask,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
