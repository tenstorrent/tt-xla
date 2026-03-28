# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wav2Vec2 model loader implementation for speech recognition (ASR).
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
    """Available Wav2Vec2 speech recognition model variants."""

    XLSR_53_ARABIC = "XLSR_53_Arabic"
    XLSR_53_RUSSIAN = "XLSR_53_Russian"
    XLSR_53_PORTUGUESE = "XLSR_53_Portuguese"
    XLSR_53_CHINESE_ZH_CN = "XLSR_53_Chinese_zh_CN"
    XLSR_53_JAPANESE = "XLSR_53_Japanese"
    XLSR_KOREAN = "XLSR_Korean"
    XLSR_HINDI = "XLSR_Hindi"
    XLSR_53_POLISH = "XLSR_53_Polish"
    XLS_R_300M_FILIPINO = "XLS_R_300M_Filipino"
    XLSR_53_DUTCH = "XLSR_53_Dutch"
    XLS_R_300M_URDU = "XLS_R_300M_Urdu"
    XLS_R_1B_NYNORSK = "XLS_R_1B_Nynorsk"
    XLS_R_300M_TURKISH = "XLS_R_300M_Turkish"
    XLSR_53_SPANISH = "XLSR_53_Spanish"
    XLSR_53_TELUGU = "XLSR_53_Telugu"


class ModelLoader(ForgeModel):
    """Wav2Vec2 model loader implementation for speech recognition."""

    _VARIANTS = {
        ModelVariant.XLSR_53_ARABIC: ModelConfig(
            pretrained_model_name="jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
        ),
        ModelVariant.XLSR_53_RUSSIAN: ModelConfig(
            pretrained_model_name="jonatasgrosman/wav2vec2-large-xlsr-53-russian",
        ),
        ModelVariant.XLSR_53_PORTUGUESE: ModelConfig(
            pretrained_model_name="jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
        ),
        ModelVariant.XLSR_53_CHINESE_ZH_CN: ModelConfig(
            pretrained_model_name="jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        ),
        ModelVariant.XLSR_53_JAPANESE: ModelConfig(
            pretrained_model_name="jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
        ),
        ModelVariant.XLSR_KOREAN: ModelConfig(
            pretrained_model_name="kresnik/wav2vec2-large-xlsr-korean",
        ),
        ModelVariant.XLSR_HINDI: ModelConfig(
            pretrained_model_name="theainerd/Wav2Vec2-large-xlsr-hindi",
        ),
        ModelVariant.XLSR_53_DUTCH: ModelConfig(
            pretrained_model_name="jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
        ),
        ModelVariant.XLSR_53_POLISH: ModelConfig(
            pretrained_model_name="jonatasgrosman/wav2vec2-large-xlsr-53-polish",
        ),
        ModelVariant.XLSR_53_HUNGARIAN: ModelConfig(
            pretrained_model_name="jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
        ),
        ModelVariant.XLS_R_300M_FILIPINO: ModelConfig(
            pretrained_model_name="Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
        ),
        ModelVariant.XLS_R_300M_URDU: ModelConfig(
            pretrained_model_name="kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
        ),
        ModelVariant.XLS_R_1B_NYNORSK: ModelConfig(
            pretrained_model_name="NbAiLab/nb-wav2vec2-1b-nynorsk",
        ),
        ModelVariant.XLS_R_300M_TURKISH: ModelConfig(
            pretrained_model_name="mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
        ),
        ModelVariant.XLSR_53_SPANISH: ModelConfig(
            pretrained_model_name="jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
        ),
        ModelVariant.XLSR_53_TELUGU: ModelConfig(
            pretrained_model_name="anuragshas/wav2vec2-large-xlsr-53-telugu",
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

        from transformers import Wav2Vec2Processor

        # Initialize processor with dtype override if specified
        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        # Load the processor
        self._processor = Wav2Vec2Processor.from_pretrained(
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

        # Load the model (from_pt=True to support variants with only PyTorch weights)
        model = FlaxWav2Vec2ForCTC.from_pretrained(
            self._variant_config.pretrained_model_name, from_pt=True, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return inputs for the Wav2Vec2 model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        import numpy as np

        # Ensure processor is initialized
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
