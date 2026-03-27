# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Distil-Whisper model loader implementation for speech recognition (ASR).
"""

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
import jax.numpy as jnp
import numpy as np


class ModelVariant(StrEnum):
    """Available Distil-Whisper speech recognition model variants."""

    DISTIL_LARGE_V3 = "Distil_Large_v3"


class ModelLoader(ForgeModel):
    """Distil-Whisper model loader implementation for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.DISTIL_LARGE_V3: ModelConfig(
            pretrained_model_name="distil-whisper/distil-large-v3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISTIL_LARGE_V3

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
            model="Distil_Whisper",
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
        """Load and return the Distil-Whisper model instance.

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

    def load_inputs(self, dtype_override=None, mesh=None):
        """Load and return sample inputs for the Distil-Whisper model.
        Args:
            dtype_override: Optional dtype to override the model's default dtype.
            mesh: Optional device mesh for sharding (DataParallel mode).
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
            return_tensors="jax",
        )
        inputs["decoder_input_ids"] = jnp.array(
            [[whisper_config.decoder_start_token_id]]
        )
        return inputs
