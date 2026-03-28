# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SAM-Audio (Segment Anything Model for Audio) loader implementation.
"""

import torch
import numpy as np
from typing import Optional
from sam_audio import SAMAudio, SAMAudioProcessor

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
    """Available SAM-Audio model variants."""

    LARGE = "Large"


class ModelLoader(ForgeModel):
    """SAM-Audio model loader implementation for audio source separation."""

    _VARIANTS = {
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="facebook/sam-audio-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="SAM_Audio",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = SAMAudioProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = SAMAudio.from_pretrained(pretrained_model_name, **kwargs)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        # Generate a synthetic audio mixture (two sine waves at different frequencies)
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio_waveform = np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 880 * t)

        # Save to a temporary wav file for processor input
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_waveform, sample_rate)
            audio_path = f.name

        description = "A tone at 440 Hz"

        inputs = self.processor(
            audios=[audio_path],
            descriptions=[description],
        )

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
