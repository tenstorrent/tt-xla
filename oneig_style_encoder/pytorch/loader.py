# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OneIG StyleEncoder model loader implementation for image feature extraction.

OneIG-StyleEncoder is a CLIP-based vision encoder that extracts normalized style
embedding vectors from images. It uses CLIPVisionModelWithProjection to produce
1280-dimensional style representations.
"""
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
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
from datasets import load_dataset


class ModelVariant(StrEnum):
    """Available OneIG StyleEncoder model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """OneIG StyleEncoder model loader implementation for image feature extraction."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="xingpng/OneIG-StyleEncoder",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="OneIG StyleEncoder",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the OneIG StyleEncoder model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The CLIPVisionModelWithProjection model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the OneIG StyleEncoder model.

        Args:
            dtype_override: Optional torch.dtype to override the input tensor's dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors containing pixel_values for the model.
        """
        if self.processor is None:
            self.processor = CLIPImageProcessor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.processor(images=image, return_tensors="pt")

        if batch_size > 1:
            inputs["pixel_values"] = inputs["pixel_values"].repeat_interleave(
                batch_size, dim=0
            )

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
