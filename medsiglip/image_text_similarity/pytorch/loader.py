# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MedSigLIP model loader implementation for image-text similarity.
"""

import torch
from transformers import AutoProcessor, AutoModel
from typing import Optional
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
    """Available MedSigLIP model variants for image-text similarity."""

    MEDSIGLIP_448 = "MedSigLIP_448"


class ModelLoader(ForgeModel):
    """MedSigLIP model loader implementation for image-text similarity tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.MEDSIGLIP_448: ModelConfig(
            pretrained_model_name="google/medsiglip-448",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MEDSIGLIP_448

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.processor = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="MedSigLIP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        # Load the processor
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MedSigLIP model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The MedSigLIP model instance for image-text similarity.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}

        # Load the model with dtype override if specified
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the MedSigLIP model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Load image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        # Define text prompts for image-text similarity
        self.text_prompts = ["a photo of 2 cats", "a photo of 2 dogs"]

        # Process both text and images
        inputs = self.processor(
            text=self.text_prompts,
            images=image,
            return_tensors="pt",
            padding="max_length",
        )

        # Replicate tensors for batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Convert the input dtype to dtype_override if specified
        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
