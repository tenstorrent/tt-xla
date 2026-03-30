# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CLIP ViT EuroSAT model loader implementation for image feature extraction.
"""

from typing import Optional

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


class ModelVariant(StrEnum):
    """Available CLIP ViT EuroSAT model variants."""

    BASE_PATCH32_EUROSAT = "Base_Patch32_EuroSAT"


class ModelLoader(ForgeModel):
    """CLIP ViT EuroSAT model loader implementation for image feature extraction."""

    _VARIANTS = {
        ModelVariant.BASE_PATCH32_EUROSAT: ModelConfig(
            pretrained_model_name="tanganke/clip-vit-base-patch32_eurosat",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_PATCH32_EUROSAT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="CLIP-ViT-EuroSAT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        from transformers import CLIPImageProcessor

        self._processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import CLIPVisionModel

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = CLIPVisionModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        from datasets import load_dataset

        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"]

        inputs = self._processor(image, return_tensors="pt")

        if batch_size > 1:
            import torch

            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
