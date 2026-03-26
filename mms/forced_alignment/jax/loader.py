# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MMS (Massively Multilingual Speech) forced alignment model loader implementation.
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


class ModelVariant(StrEnum):
    """Available MMS forced alignment model variants."""

    MMS_300M_1130 = "MMS_300M_1130"


class ModelLoader(ForgeModel):
    """MMS forced alignment model loader implementation."""

    _VARIANTS = {
        ModelVariant.MMS_300M_1130: ModelConfig(
            pretrained_model_name="MahmoudAshraf/mms-300m-1130-forced-aligner",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MMS_300M_1130

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

        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MMS",
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

        from transformers import Wav2Vec2Processor

        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        self._processor = Wav2Vec2Processor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MMS forced alignment model instance.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """
        from transformers import FlaxWav2Vec2ForCTC

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = FlaxWav2Vec2ForCTC.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return inputs for the MMS forced alignment model.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        import numpy as np

        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="jax",
        )

        return inputs
