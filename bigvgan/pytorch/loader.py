# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BigVGAN v2 Neural Vocoder model loader implementation.

BigVGAN is a universal neural vocoder that generates high-quality audio
waveforms from mel spectrograms.
"""
import json

import torch
from huggingface_hub import hf_hub_download
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
    """Available BigVGAN model variants."""

    V2_22KHZ_80BAND_256X = "v2_22khz_80band_256x"
    V2_44KHZ_128BAND_512X = "v2_44khz_128band_512x"


class ModelLoader(ForgeModel):
    """BigVGAN v2 Neural Vocoder model loader implementation."""

    _VARIANTS = {
        ModelVariant.V2_22KHZ_80BAND_256X: ModelConfig(
            pretrained_model_name="nvidia/bigvgan_v2_22khz_80band_256x",
        ),
        ModelVariant.V2_44KHZ_128BAND_512X: ModelConfig(
            pretrained_model_name="nvidia/bigvgan_v2_44khz_128band_512x",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2_22KHZ_80BAND_256X

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="bigvgan",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BigVGAN model instance.

        Manually downloads config and weights from HuggingFace Hub to avoid
        compatibility issues between the bigvgan package and huggingface_hub.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The BigVGAN vocoder model instance.
        """
        from bigvgan.bigvgan import BigVGAN
        from bigvgan.env import AttrDict

        repo_id = self._variant_config.pretrained_model_name

        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path) as f:
            h = AttrDict(json.load(f))

        model = BigVGAN(h, use_cuda_kernel=False)

        weights_path = hf_hub_download(repo_id=repo_id, filename="bigvgan_generator.pt")
        checkpoint = torch.load(weights_path, map_location="cpu")

        try:
            model.load_state_dict(checkpoint["generator"])
        except RuntimeError:
            model.remove_weight_norm()
            model.load_state_dict(checkpoint["generator"])

        model.remove_weight_norm()
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the BigVGAN model.

        The vocoder expects a mel spectrogram tensor with shape
        (batch, mel_bins, seq_len).

        Args:
            dtype_override: Optional torch.dtype to override the input tensor's dtype.

        Returns:
            torch.Tensor: Mel spectrogram tensor.
        """
        mel_bins_map = {
            ModelVariant.V2_22KHZ_80BAND_256X: 80,
            ModelVariant.V2_44KHZ_128BAND_512X: 128,
        }
        mel_bins = mel_bins_map[self._variant]
        # Shape: (batch_size, num_mel_bins, sequence_length)
        mel = torch.randn(1, mel_bins, 256)

        if dtype_override is not None:
            mel = mel.to(dtype_override)

        return mel
