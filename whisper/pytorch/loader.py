# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper model loader implementation
"""

import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperModel,
    AutoFeatureExtractor,
)
from datasets import load_dataset
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
    WHISPER_LARGE_V3 = "openai/whisper-large-v3"
    WHISPER_LARGE_V3_TURBO = "openai/whisper-large-v3-turbo"


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
        ModelVariant.WHISPER_LARGE_V3: ModelConfig(
            pretrained_model_name="openai/whisper-large-v3",
        ),
        ModelVariant.WHISPER_LARGE_V3_TURBO: ModelConfig(
            pretrained_model_name="openai/whisper-large-v3-turbo",
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
            group=(
                ModelGroup.RED
                if variant == ModelVariant.WHISPER_LARGE_V3
                else ModelGroup.GENERALITY
            ),
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
        self.feature_extractor = None

    def load_model(self, dtype_override=None):
        """Load a Whisper model from Hugging Face."""

        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Common model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self._variant == ModelVariant.WHISPER_LARGE_V3:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                pretrained_model_name
            )
            model = WhisperModel.from_pretrained(pretrained_model_name, **model_kwargs)
        else:
            processor_kwargs = {}
            if dtype_override is not None:
                processor_kwargs["torch_dtype"] = dtype_override

            self.processor = WhisperProcessor.from_pretrained(
                pretrained_model_name, use_cache=False, **processor_kwargs
            )
            model = WhisperForConditionalGeneration.from_pretrained(
                pretrained_model_name, use_cache=False, **model_kwargs
            )

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for Whisper model."""

        if self._variant == ModelVariant.WHISPER_LARGE_V3:

            if self.feature_extractor is None:
                self.load_model()  # This will initialize the feature_extractor

            ds = load_dataset(
                "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
            )
            input_audio = self.feature_extractor(
                ds[0]["audio"]["array"], return_tensors="pt"
            )
            input_features = input_audio.input_features

            # Convert to the specified dtype if provided
            if dtype_override is not None:
                input_features = input_features.to(dtype_override)

            return input_features
        else:

            if self.processor is None:
                self.load_model()  # This will initialize the processor

            weights_pth = get_file("test_files/pytorch/whisper/1272-128104-0000.pt")
            sample = torch.load(weights_pth, weights_only=False)
            sample_audio = sample["audio"]["array"]

            inputs = self.processor(sample_audio, return_tensors="pt")
            input_features = inputs.input_features

            # Convert to the specified dtype if provided
            if dtype_override is not None:
                input_features = input_features.to(dtype_override)

            # Create decoder_input_ids starting with the decoder start token
            # For WhisperForConditionalGeneration, we need to provide decoder_input_ids
            decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                "<|startoftranscript|>"
            )
            decoder_input_ids = torch.tensor(
                [[decoder_start_token_id]], dtype=torch.long
            )

            return {
                "input_features": input_features,
                "decoder_input_ids": decoder_input_ids,
            }
