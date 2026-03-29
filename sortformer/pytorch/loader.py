# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
NVIDIA Streaming Sortformer speaker diarization model loader implementation using PyTorch.

Loads the SortformerEncLabelModel for end-to-end speaker diarization,
identifying and labeling individual speakers in multi-speaker audio.
"""

import torch
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
    """Available Sortformer speaker diarization model variants."""

    STREAMING_4SPK_V2_1 = "Streaming_4spk_v2.1"


class ModelLoader(ForgeModel):
    """NVIDIA Streaming Sortformer speaker diarization model loader (PyTorch)."""

    _VARIANTS = {
        ModelVariant.STREAMING_4SPK_V2_1: ModelConfig(
            pretrained_model_name="nvidia/diar_streaming_sortformer_4spk-v2.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STREAMING_4SPK_V2_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Sortformer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from nemo.collections.asr.models import SortformerEncLabelModel

        model = SortformerEncLabelModel.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        # Generate a synthetic 1-second mono audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        input_signal = torch.tensor(audio_array).unsqueeze(0)
        input_signal_length = torch.tensor([len(audio_array)])

        if dtype_override is not None:
            input_signal = input_signal.to(dtype_override)

        return {
            "input_signal": input_signal,
            "input_signal_length": input_signal_length,
        }
