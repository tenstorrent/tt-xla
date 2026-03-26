# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BigVGAN v2 Neural Vocoder model loader implementation.

BigVGAN is a universal neural vocoder that generates high-quality audio
waveforms from mel spectrograms.
"""
import torch
import bigvgan as bigvgan_module
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


class ModelLoader(ForgeModel):
    """BigVGAN v2 Neural Vocoder model loader implementation."""

    _VARIANTS = {
        ModelVariant.V2_22KHZ_80BAND_256X: ModelConfig(
            pretrained_model_name="nvidia/bigvgan_v2_22khz_80band_256x",
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

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The BigVGAN vocoder model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = bigvgan_module.BigVGAN.from_pretrained(
            pretrained_model_name, use_cuda_kernel=False
        )
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
            torch.Tensor: Mel spectrogram tensor with shape (1, 80, 256).
        """
        # Shape: (batch_size, num_mel_bins, sequence_length)
        mel = torch.randn(1, 80, 256)

        if dtype_override is not None:
            mel = mel.to(dtype_override)

        return mel
