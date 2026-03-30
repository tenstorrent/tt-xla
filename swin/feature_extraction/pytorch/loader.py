# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Swin feature extraction model loader implementation.
"""

from transformers import SwinModel, AutoImageProcessor
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
    """Available Swin feature extraction model variants."""

    TINY_RANDOM_PATCH4_WINDOW7_224 = "Tiny_Random_Patch4_Window7_224"


class ModelLoader(ForgeModel):
    """Swin feature extraction model loader implementation."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM_PATCH4_WINDOW7_224: ModelConfig(
            pretrained_model_name="yujiepan/tiny-random-swin-patch4-window7-224",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM_PATCH4_WINDOW7_224

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
            model="Swin",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Swin feature extraction model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Swin model instance.
        """
        model_name = self._variant_config.pretrained_model_name

        model = SwinModel.from_pretrained(model_name, **kwargs)
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
            self._processor = AutoImageProcessor.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"]

        inputs = self._processor(image, return_tensors="pt")

        return inputs
