# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EoMT model loader implementation for panoptic segmentation tasks.
"""

import torch
from typing import Optional
from transformers import EomtForUniversalSegmentation, AutoImageProcessor

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
from datasets import load_dataset


class ModelVariant(StrEnum):
    """Available EoMT model variants for panoptic segmentation."""

    LARGE_640_COCO_PANOPTIC = "Large_640_Coco_Panoptic"


class ModelLoader(ForgeModel):
    """EoMT model loader implementation for panoptic segmentation tasks."""

    _VARIANTS = {
        ModelVariant.LARGE_640_COCO_PANOPTIC: ModelConfig(
            pretrained_model_name="tue-mps/coco_panoptic_eomt_large_640"
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_640_COCO_PANOPTIC

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.image_processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="EoMT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_PANOPTIC_SEG,
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
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = EomtForUniversalSegmentation.from_pretrained(
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
                if dtype_override is not None:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
