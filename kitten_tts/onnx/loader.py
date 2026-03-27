# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KittenTTS model loader implementation for text-to-speech tasks.
"""
import os

import numpy as np
import onnx
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
    """Available KittenTTS model variants."""

    KITTEN_TTS_NANO_0_8_FP32 = "nano-0.8-fp32"


class ModelLoader(ForgeModel):
    """KittenTTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.KITTEN_TTS_NANO_0_8_FP32: ModelConfig(
            pretrained_model_name="KittenML/kitten-tts-nano-0.8-fp32",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KITTEN_TTS_NANO_0_8_FP32

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model_dir = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="KittenTTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        from huggingface_hub import snapshot_download

        self._model_dir = snapshot_download(
            repo_id=self._variant_config.pretrained_model_name
        )
        model = onnx.load(os.path.join(self._model_dir, "kitten_tts_nano_v0_8.onnx"))
        return model

    def load_inputs(self, **kwargs):
        # Dummy phoneme token IDs: [pad, ...tokens..., end_marker, pad]
        input_ids = torch.tensor([[0, 47, 15, 92, 33, 71, 10, 0]], dtype=torch.int64)
        # Load a voice embedding from voices.npz
        voices = np.load(os.path.join(self._model_dir, "voices.npz"))
        voice_key = "expr-voice-2-m"  # Jasper voice
        voice_data = voices[voice_key]
        style = torch.tensor(voice_data[:1], dtype=torch.float32)
        speed = torch.tensor([1.0], dtype=torch.float32)
        return input_ids, style, speed
