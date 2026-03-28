# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ViT feature extraction model loader implementation.
"""

from transformers import ViTModel, ViTImageProcessor
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
    """Available ViT feature extraction model variants."""

    HUGE_PATCH14_224_IN_21K = "Huge_Patch14_224_In_21K"


class ModelLoader(ForgeModel):
    """ViT feature extraction model loader implementation."""

    _VARIANTS = {
        ModelVariant.HUGE_PATCH14_224_IN_21K: ModelConfig(
            pretrained_model_name="google/vit-huge-patch14-224-in21k",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUGE_PATCH14_224_IN_21K

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ViT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the ViT feature extraction model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The ViT model instance.
        """
        model_name = self._variant_config.pretrained_model_name

        model = ViTModel.from_pretrained(model_name, **kwargs)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            dict: Preprocessed input tensors.
        """
        from datasets import load_dataset

        if self._processor is None:
            self._processor = ViTImageProcessor.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"]

        inputs = self._processor(image, return_tensors="pt")

        return inputs
