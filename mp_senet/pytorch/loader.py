# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MP-SENet speech enhancement model loader implementation.

MP-SENet is a speech enhancement network that performs parallel magnitude
and phase spectra denoising in the time-frequency domain.
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
    """Available MP-SENet model variants."""

    VB = "VB"


class ModelLoader(ForgeModel):
    """MP-SENet speech enhancement model loader implementation."""

    _VARIANTS = {
        ModelVariant.VB: ModelConfig(
            pretrained_model_name="JacobLinCool/MP-SENet-VB",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VB

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="mp_senet",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from MPSENet import MPSENet

        model = MPSENet.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        n_fft = 400
        hop_size = 100
        win_size = 400
        compress_factor = 0.3

        # Generate a synthetic noisy audio waveform (1 second at 16kHz)
        sample_rate = 16000
        waveform = torch.randn(1, sample_rate)

        # Convert to magnitude and phase via STFT
        hann_window = torch.hann_window(win_size)
        stft_spec = torch.stft(
            waveform,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        stft_spec = torch.view_as_real(stft_spec)
        mag = torch.sqrt(stft_spec.pow(2).sum(-1) + 1e-9)
        pha = torch.atan2(stft_spec[:, :, :, 1] + 1e-10, stft_spec[:, :, :, 0] + 1e-5)
        mag = torch.pow(mag, compress_factor)

        # Model expects [B, F, T]
        noisy_amp = mag
        noisy_pha = pha

        if dtype_override is not None:
            noisy_amp = noisy_amp.to(dtype_override)
            noisy_pha = noisy_pha.to(dtype_override)

        return (noisy_amp, noisy_pha)
