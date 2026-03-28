# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SkyrimLikeVoices VITS2 text-to-speech model loader implementation.

This is a multi-speaker VITS2 model trained on Skyrim game character voices,
supporting 123 distinct speaker identities.
"""
import json

import torch
import torch.nn as nn
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


class VitsSynthesizerWrapper(nn.Module):
    """Wrapper around VITS2 SynthesizerTrn for a clean forward pass.

    Exposes the VITS2 synthesizer inference as a standard forward method
    that takes phoneme token IDs, lengths, and speaker IDs to produce
    audio waveforms.
    """

    def __init__(self, synthesizer):
        super().__init__()
        self.synthesizer = synthesizer

    def forward(self, x, x_lengths, sid):
        return self.synthesizer.infer(
            x,
            x_lengths,
            sid=sid,
            noise_scale=0.667,
            length_scale=1.0,
            noise_scale_w=0.8,
        )


class ModelVariant(StrEnum):
    """Available SkyrimLikeVoices model variants."""

    SKYRIM_LIKE_VOICES = "skyrim_like_voices"


class ModelLoader(ForgeModel):
    """SkyrimLikeVoices VITS2 text-to-speech model loader implementation."""

    _VARIANTS = {
        ModelVariant.SKYRIM_LIKE_VOICES: ModelConfig(
            pretrained_model_name="tylermaister/SkyrimLikeVoices",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SKYRIM_LIKE_VOICES

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="skyrim_like_voices",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SkyrimLikeVoices VITS2 model.

        Downloads config and checkpoint from HuggingFace Hub, instantiates
        the VITS2 SynthesizerTrn model, and loads the pretrained weights.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: Wrapped VITS2 synthesizer model.
        """
        from melo.models import SynthesizerTrn
        from melo.utils import get_hparams_from_file

        repo_id = self._variant_config.pretrained_model_name

        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename="checkpoint.pth")

        hps = get_hparams_from_file(config_path)

        model = SynthesizerTrn(
            len(hps.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return VitsSynthesizerWrapper(model)

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the VITS2 model.

        Returns phoneme token IDs, sequence lengths, and a speaker ID
        suitable for inference.

        Args:
            dtype_override: Optional torch.dtype (not used for integer inputs).

        Returns:
            tuple: (x, x_lengths, sid) where x is phoneme token IDs,
                   x_lengths is sequence lengths, and sid is speaker ID.
        """
        # Phoneme token IDs: [batch, seq_len]
        x = torch.randint(0, 200, (1, 50), dtype=torch.long)
        # Sequence lengths: [batch]
        x_lengths = torch.tensor([50], dtype=torch.long)
        # Speaker ID (femaleargonian=0): [batch]
        sid = torch.tensor([0], dtype=torch.long)
        return x, x_lengths, sid
