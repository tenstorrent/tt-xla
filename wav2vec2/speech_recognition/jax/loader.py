# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wav2Vec2 model loader implementation for speech recognition (ASR).
"""

from typing import Optional

import datasets
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
    """Available Wav2Vec2 speech recognition model variants."""

    XLSR_53_RUSSIAN = "XLSR_53_Russian"


class ModelLoader(ForgeModel):
    """Wav2Vec2 model loader implementation for speech recognition."""

    _VARIANTS = {
        ModelVariant.XLSR_53_RUSSIAN: ModelConfig(
            pretrained_model_name="jonatasgrosman/wav2vec2-large-xlsr-53-russian",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XLSR_53_RUSSIAN

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
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        # Use the provided variant or fall back to default
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Wav2Vec2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_processor(self, dtype_override=None):
        """Load audio processor for the current variant.

        Args:
            dtype_override: Optional dtype to override the processor's default dtype.

        Returns:
            processor: The loaded audio processor instance
        """

        from transformers import AutoProcessor

        # Initialize processor with dtype override if specified
        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        # Load the processor
        self._processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Wav2Vec2 model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """
        from transformers import FlaxWav2Vec2ForCTC

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        # Load the model
        model = FlaxWav2Vec2ForCTC.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return inputs for the Wav2Vec2 model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        from datasets import load_dataset

        # Force datasets to use soundfile backend instead of torchcodec
        datasets.config.AUDIO_BACKEND = "soundfile"

        # Ensure processor is initialized
        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        sample = dataset[0]["audio"]
        inputs = self._processor(
            sample["array"],
            sampling_rate=sample["sampling_rate"],
            return_tensors="jax",
        )

        return inputs
