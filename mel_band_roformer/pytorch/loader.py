# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mel-Band RoFormer model loader for audio source separation.

Mel-Band RoFormer is a transformer-based model for music source separation
that uses mel-band frequency decomposition with rotary position embeddings.
"""

from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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
    """Available Mel-Band RoFormer model variants."""

    FP16 = "fp16"
    FP32 = "fp32"


class ModelLoader(ForgeModel):
    """Mel-Band RoFormer model loader for audio source separation."""

    _VARIANTS = {
        ModelVariant.FP16: ModelConfig(
            pretrained_model_name="Kijai/MelBandRoFormer_comfy",
        ),
        ModelVariant.FP32: ModelConfig(
            pretrained_model_name="Kijai/MelBandRoFormer_comfy",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FP16

    _MODEL_FILES = {
        ModelVariant.FP16: "MelBandRoformer_fp16.safetensors",
        ModelVariant.FP32: "MelBandRoformer_fp32.safetensors",
    }

    _MODEL_CONFIG = {
        "dim": 384,
        "depth": 6,
        "stereo": True,
        "num_stems": 1,
        "time_transformer_depth": 1,
        "freq_transformer_depth": 1,
        "num_bands": 60,
        "dim_head": 64,
        "heads": 8,
        "attn_dropout": 0,
        "ff_dropout": 0,
        "flash_attn": False,
        "dim_freqs_in": 1025,
        "sample_rate": 44100,
        "stft_n_fft": 2048,
        "stft_hop_length": 441,
        "stft_win_length": 2048,
        "stft_normalized": False,
        "mask_estimator_depth": 2,
        "multi_stft_resolution_loss_weight": 1.0,
        "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
        "multi_stft_hop_size": 147,
        "multi_stft_normalized": False,
    }

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MelBandRoFormer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from .src.mel_band_roformer import MelBandRoformer

        model = MelBandRoformer(**self._MODEL_CONFIG)

        repo_id = self._variant_config.pretrained_model_name
        filename = self._MODEL_FILES[self._variant]
        weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
        state_dict = load_file(weight_path)
        model.load_state_dict(state_dict, strict=True)

        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        # Generate synthetic stereo audio at 44100Hz (1 second)
        sample_rate = 44100
        duration_seconds = 1
        audio = torch.randn(1, 2, sample_rate * duration_seconds)

        if dtype_override is not None:
            audio = audio.to(dtype_override)

        return audio
