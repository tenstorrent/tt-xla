# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SpeechBrain VAD CRDNN LibriParty model loader for voice activity detection.
"""

from typing import Optional

import torch

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available SpeechBrain VAD CRDNN model variants."""

    LIBRIPARTY = "LibriParty"


class VADCRDNNModel(torch.nn.Module):
    """Wrapper module for the SpeechBrain VAD CRDNN pipeline."""

    def __init__(self, vad):
        super().__init__()
        self.compute_features = vad.mods.compute_features
        self.mean_var_norm = vad.mods.mean_var_norm
        self.crdnn = vad.mods.crdnn
        self.output_layer = vad.mods.output

    def forward(self, wavs, wav_lens):
        feats = self.compute_features(wavs)
        feats = self.mean_var_norm(feats, wav_lens)
        outputs = self.crdnn(feats)
        outputs = self.output_layer(outputs)
        return outputs


class ModelLoader(ForgeModel):
    """SpeechBrain VAD CRDNN LibriParty model loader."""

    _VARIANTS = {
        ModelVariant.LIBRIPARTY: ModelConfig(
            pretrained_model_name="speechbrain/vad-crdnn-libriparty",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LIBRIPARTY

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="VADCRDNNLibriParty",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the SpeechBrain VAD CRDNN model."""
        from speechbrain.inference.VAD import VAD

        vad = VAD.from_hparams(
            source=self._variant_config.pretrained_model_name,
            **kwargs,
        )

        model = VADCRDNNModel(vad)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Generate synthetic 1-second audio waveform at 16kHz."""
        waveform = torch.randn(1, 16000)
        wav_lens = torch.tensor([1.0])

        if dtype_override is not None:
            waveform = waveform.to(dtype_override)
            wav_lens = wav_lens.to(dtype_override)

        return [waveform, wav_lens]
