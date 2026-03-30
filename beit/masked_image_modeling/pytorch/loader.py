# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BEiT model loader implementation for masked image modeling.
"""

from transformers import BeitForMaskedImageModeling, BeitImageProcessor
from datasets import load_dataset
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
    """Available BEiT model variants for masked image modeling."""

    BASE_PATCH16_224_PT22K = "Base_Patch16_224_pt22k"


class ModelLoader(ForgeModel):
    """BEiT model loader implementation for masked image modeling."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE_PATCH16_224_PT22K: ModelConfig(
            pretrained_model_name="microsoft/beit-base-patch16-224-pt22k",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE_PATCH16_224_PT22K

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

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
            model="BEiT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load BEiT model for masked image modeling from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The BEiT model instance for masked image modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Initialize processor
        self.processor = BeitImageProcessor.from_pretrained(pretrained_model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BeitForMaskedImageModeling.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the BEiT masked image modeling model.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self.load_model(dtype_override=dtype_override)

        # Load image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt").pixel_values

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
