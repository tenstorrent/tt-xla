# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma model loader implementation for text extraction and translation task.
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Gemma model variants."""

    TRANSLATEGEMMA_4B_IT = "translategemma_4b_it"


class ModelLoader(ForgeModel):
    """Gemma model loader implementation for text extraction and translation task."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.TRANSLATEGEMMA_4B_IT: LLMModelConfig(
            pretrained_model_name="google/translategemma-4b-it",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.TRANSLATEGEMMA_4B_IT

    # Sample data for text extraction and translation
    sample_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source_lang_code": "cs",
                    "target_lang_code": "de-DE",
                    "url": "https://c7.alamy.com/comp/2YAX36N/traffic-signs-in-czech-republic-pedestrian-zone-2YAX36N.jpg",
                },
            ],
        }
    ]

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
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="gemma",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load processor for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the processor's default dtype.

        Returns:
            The loaded processor instance
        """
        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self.processor

    def load_model(self, dtype_override=None):
        """Load and return the Gemma model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, uses the model's default dtype from HuggingFace.

        Returns:
            torch.nn.Module: The Gemma model instance for text extraction and translation.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Gemma model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
                           Only float32 tensors will be converted to the specified dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        inputs = self.processor.apply_chat_template(
            self.sample_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
