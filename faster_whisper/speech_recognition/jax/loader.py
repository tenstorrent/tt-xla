# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Faster Whisper model loader implementation for speech recognition (ASR).

Note: Faster Whisper models are CTranslate2-quantized versions of OpenAI's
Whisper models. Since CTranslate2 format is not compatible with JAX/Flax,
this loader uses the base OpenAI Whisper models via
FlaxWhisperForConditionalGeneration.
"""

from typing import Optional

import jax.numpy as jnp
import numpy as np

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Faster Whisper speech recognition model variants."""

    TINY_INT8 = "Tiny_int8"
    LARGE_V2 = "Large_v2"
    LARGE_V3_TURBO = "Large_v3_Turbo"


class ModelLoader(ForgeModel):
    """Faster Whisper model loader implementation for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.TINY_INT8: ModelConfig(
            pretrained_model_name="openai/whisper-tiny",
        ),
        ModelVariant.LARGE_V2: ModelConfig(
            pretrained_model_name="openai/whisper-large-v2",
        ),
        ModelVariant.LARGE_V3_TURBO: ModelConfig(
            pretrained_model_name="openai/whisper-large-v3-turbo",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_INT8

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
        """Method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Faster_Whisper",
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

        from transformers import WhisperProcessor

        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        self._processor = WhisperProcessor.from_pretrained(
            self._model_name, **processor_kwargs
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Whisper model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """

        from transformers import FlaxWhisperForConditionalGeneration

        from ....tools.jax_utils import cast_hf_model_to_type

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = FlaxWhisperForConditionalGeneration.from_pretrained(
            self._model_name, from_pt=True, **model_kwargs
        )
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Whisper model.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        from transformers import WhisperConfig

        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Generate synthetic 30-second audio at 16kHz to match Whisper's receptive field
        sampling_rate = 16000
        duration_seconds = 30
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        whisper_config = WhisperConfig.from_pretrained(self._model_name)

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="np",
        )
        inputs = {k: jnp.array(v) for k, v in inputs.items()}
        inputs["decoder_input_ids"] = jnp.array(
            [[whisper_config.decoder_start_token_id]]
        )
        return inputs
