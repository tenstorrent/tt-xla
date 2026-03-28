# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ViT feature extraction model loader implementation for PyTorch.
"""

from typing import Optional

from transformers import ViTModel, ViTImageProcessor
from datasets import load_dataset

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

    BASE_PATCH32_224_IN_21K = "Base_Patch32_224_In_21K"


class ModelLoader(ForgeModel):
    """ViT feature extraction model loader implementation for PyTorch."""

    _VARIANTS = {
        ModelVariant.BASE_PATCH32_224_IN_21K: ModelConfig(
            pretrained_model_name="google/vit-base-patch32-224-in21k",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_PATCH32_224_IN_21K

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._processor = None
        self._model_name = self._variant_config.pretrained_model_name

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
            model: The loaded ViT model instance
        """
        model = ViTModel.from_pretrained(self._model_name, **kwargs)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).
            image: Optional input image. If None, loads from HuggingFace datasets.

        Returns:
            dict: Preprocessed inputs with pixel_values tensor.
        """
        if self._processor is None:
            self._processor = ViTImageProcessor.from_pretrained(self._model_name)

        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]

        inputs = self._processor(image, return_tensors="pt")

        return inputs
