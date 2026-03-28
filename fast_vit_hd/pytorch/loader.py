# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FastViT-HD image encoder model loader implementation.

Loads the FastViT-HD vision backbone from kevin510/fast-vit-hd, which is
Apple's FastViT-HD image encoder from the FastVLM paper (arXiv:2412.13303).
Produces per-image patch embeddings of shape (1, 256, 3072) for 1024x1024 input.
"""

import torch
from transformers import AutoModel, AutoImageProcessor
from datasets import load_dataset
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
    """Available FastViT-HD model variants."""

    HD = "Hd"


class ModelLoader(ForgeModel):
    """FastViT-HD image encoder model loader for image feature extraction (PyTorch)."""

    _VARIANTS = {
        ModelVariant.HD: ModelConfig(
            pretrained_model_name="kevin510/fast-vit-hd",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HD

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="FastViT-HD",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load image processor for the current variant."""
        self.processor = AutoImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FastViT-HD image encoder model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The FastViT-HD model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the FastViT-HD model.

        Args:
            dtype_override: Optional torch.dtype to override input dtype.
            batch_size: Batch size for the inputs.

        Returns:
            torch.Tensor: Preprocessed pixel_values tensor.
        """
        if self.processor is None:
            self._load_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.processor(
            images=image,
            do_center_crop=False,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"]

        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
