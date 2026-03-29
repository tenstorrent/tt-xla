# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Audio Spectrogram Transformer (AST) model loader implementation for audio classification.
"""

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
    """Available AST audio classification model variants."""

    FINETUNED_AUDIOSET_16_16_0442 = "Finetuned_Audioset_16_16_0.442"
    DISTIL_AUDIOSET = "Distil_Audioset"


class ModelLoader(ForgeModel):
    """AST model loader implementation for audio classification (PyTorch)."""

    _VARIANTS = {
        ModelVariant.FINETUNED_AUDIOSET_16_16_0442: ModelConfig(
            pretrained_model_name="MIT/ast-finetuned-audioset-16-16-0.442",
        ),
        ModelVariant.DISTIL_AUDIOSET: ModelConfig(
            pretrained_model_name="bookbot/distil-ast-audioset",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FINETUNED_AUDIOSET_16_16_0442

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="AST",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import ASTFeatureExtractor

        self._processor = ASTFeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import ASTForAudioClassification

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = ASTForAudioClassification.from_pretrained(
            self._variant_config.pretrained_model_name,
            **model_kwargs,
        )
        model.eval()

        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        if self._processor is None:
            self._load_processor()

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        return inputs
