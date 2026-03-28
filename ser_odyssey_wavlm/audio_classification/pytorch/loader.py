# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
SER-Odyssey-Baseline-WavLM-Categorical model loader for audio classification
(speech emotion recognition).
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
    """Available SER-Odyssey WavLM audio classification model variants."""

    BASELINE_CATEGORICAL = "Baseline Categorical"


class ModelLoader(ForgeModel):
    """SER-Odyssey WavLM model loader for audio classification (PyTorch)."""

    _VARIANTS = {
        ModelVariant.BASELINE_CATEGORICAL: ModelConfig(
            pretrained_model_name="3loi/SER-Odyssey-Baseline-WavLM-Categorical",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASELINE_CATEGORICAL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SER-Odyssey-WavLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForAudioClassification

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForAudioClassification.from_pretrained(
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
        import torch

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        raw_wav = np.random.randn(sampling_rate * duration_seconds).astype(np.float32)

        mask = torch.ones(1, len(raw_wav))
        wavs = torch.tensor(raw_wav).unsqueeze(0)

        return {"x": wavs, "mask": mask}
