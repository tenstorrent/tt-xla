# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
WavLM model loader implementation for speaker verification (x-vector).
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
    """Available WavLM speaker verification model variants."""

    BASE_SV = "Base SV"


class ModelLoader(ForgeModel):
    """WavLM model loader implementation for speaker verification (x-vector)."""

    _VARIANTS = {
        ModelVariant.BASE_SV: ModelConfig(
            pretrained_model_name="microsoft/wavlm-base-sv",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_SV

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._feature_extractor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="WavLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_XVECTOR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_feature_extractor(self, dtype_override=None):
        from transformers import AutoFeatureExtractor

        feature_extractor_kwargs = {}
        if dtype_override is not None:
            feature_extractor_kwargs["dtype"] = dtype_override

        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name, **feature_extractor_kwargs
        )

        return self._feature_extractor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import WavLMForXVector

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = WavLMForXVector.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        if self._feature_extractor is None:
            self._load_feature_extractor(dtype_override=dtype_override)

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        return inputs
