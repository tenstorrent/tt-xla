# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stereo model loader implementation for music generation
"""
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
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
    """Available Stereo model variants."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class ModelLoader(ForgeModel):
    """Stereo model loader implementation for music generation tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.SMALL: ModelConfig(
            pretrained_model_name="facebook/musicgen-small",
        ),
        ModelVariant.MEDIUM: ModelConfig(
            pretrained_model_name="facebook/musicgen-medium",
        ),
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="facebook/musicgen-large",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SMALL

    # Shared configuration parameters
    sample_text = (
        "80s pop track with bassy drums and synth",
        "90s rock song with loud guitars and heavy drums",
    )

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
            model="stereo",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
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
        # Initialize processor with dtype override if specified
        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["torch_dtype"] = dtype_override

        # Load the processor
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self.processor

    def load_model(self, dtype_override=None):
        """Load and return the Stereo model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Stereo model instance for music generation.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure processor is loaded
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = MusicgenForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        self.model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Stereo model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        inputs = self.processor(
            text=[self.sample_text],
            padding=True,
            return_tensors="pt",
        )

        pad_token_id = self.model.generation_config.pad_token_id
        decoder_input_ids = (
            torch.ones(
                (inputs.input_ids.shape[0] * self.model.decoder.num_codebooks, 1),
                dtype=torch.long,
            )
            * pad_token_id
        )

        inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
        }
        return inputs
