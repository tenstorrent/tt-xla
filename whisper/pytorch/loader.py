# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper model loader implementation
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ...tools.utils import get_file
from ...base import ForgeModel
from typing import Optional


class ModelVariant(StrEnum):
    """Available Whisper model variants."""

    WHISPER_TINY = "openai/whisper-tiny"
    WHISPER_BASE = "openai/whisper-base"
    WHISPER_SMALL = "openai/whisper-small"
    WHISPER_MEDIUM = "openai/whisper-medium"
    WHISPER_LARGE = "openai/whisper-large"


class ModelLoader(ForgeModel):
    """Whisper model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.WHISPER_TINY: ModelConfig(
            pretrained_model_name="openai/whisper-tiny",
        ),
        ModelVariant.WHISPER_BASE: ModelConfig(
            pretrained_model_name="openai/whisper-base",
        ),
        ModelVariant.WHISPER_SMALL: ModelConfig(
            pretrained_model_name="openai/whisper-small",
        ),
        ModelVariant.WHISPER_MEDIUM: ModelConfig(
            pretrained_model_name="openai/whisper-medium",
        ),
        ModelVariant.WHISPER_LARGE: ModelConfig(
            pretrained_model_name="openai/whisper-large",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.WHISPER_TINY

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
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
            model="whisper",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.processor = None

    def load_model(self, dtype_override=None):
        """Load a Whisper model from Hugging Face."""

        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name
        # Initialize processor first with default or overridden dtype
        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["torch_dtype"] = dtype_override

        self.processor = WhisperProcessor.from_pretrained(
            pretrained_model_name,
            use_cache=False,
            return_dict=False,
            **processor_kwargs
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name, use_cache=False, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self):
        """Generate sample inputs for Whisper model."""

        # Ensure processor is initialized
        if self.processor is None:
            self.load_model()  # This will initialize the processor
        weights_pth = get_file("test_files/pytorch/whisper/1272-128104-0000.pt")
        sample = torch.load(weights_pth, weights_only=False)
        sample_audio = sample["audio"]["array"]

        inputs = self.processor(sample_audio, return_tensors="pt")
        input_features = inputs.input_features

        return input_features
