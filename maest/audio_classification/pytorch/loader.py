# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MAEST model loader implementation for audio classification.
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
    """Available MAEST audio classification model variants."""

    DISCOGS_30S_PW_129E_519L = "Discogs_30s_pw_129e_519l"


class ModelLoader(ForgeModel):
    """MAEST model loader implementation for audio classification (PyTorch)."""

    _VARIANTS = {
        ModelVariant.DISCOGS_30S_PW_129E_519L: ModelConfig(
            pretrained_model_name="mtg-upf/discogs-maest-30s-pw-129e-519l",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISCOGS_30S_PW_129E_519L

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MAEST",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        from transformers import AutoFeatureExtractor

        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        self._processor = AutoFeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **processor_kwargs,
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
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
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
            return_tensors="pt",
        )

        return inputs
