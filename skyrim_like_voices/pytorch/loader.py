# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SkyrimLikeVoices VITS2 text-to-speech model loader implementation.

This is a multi-speaker VITS2 model trained on Skyrim game character voices,
supporting 123 distinct speaker identities. Uses the MeloTTS SynthesizerTrn
architecture.
"""
import json
import os
import sys

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


class HParams:
    """Lightweight hyperparameter container that supports attribute access.

    Replaces melo.utils.get_hparams_from_file to avoid pulling in
    librosa and other heavy transitive dependencies.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


class VitsSynthesizerWrapper(nn.Module):
    """Wrapper around VITS2 SynthesizerTrn for a clean forward pass.

    Exposes the VITS2 synthesizer inference as a standard forward method
    that takes phoneme token IDs, lengths, speaker ID, tone, language,
    and BERT embeddings to produce audio waveforms.
    """

    def __init__(self, synthesizer):
        super().__init__()
        self.synthesizer = synthesizer

    def forward(self, x, x_lengths, sid, tone, language, bert, ja_bert):
        return self.synthesizer.infer(
            x,
            x_lengths,
            sid,
            tone,
            language,
            bert,
            ja_bert,
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
        self._hps = None

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
        import melo

        melo_dir = os.path.dirname(melo.__file__)
        if melo_dir not in sys.path:
            sys.path.insert(0, melo_dir)

        from melo.models import SynthesizerTrn

        repo_id = self._variant_config.pretrained_model_name

        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename="checkpoint.pth")

        with open(config_path) as f:
            data = json.load(f)
        hps = HParams(**data)
        self._hps = hps

        model = SynthesizerTrn(
            len(hps.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **dict(hps.model.items()),
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return VitsSynthesizerWrapper(model)

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the VITS2 model.

        Returns phoneme token IDs, sequence lengths, speaker ID, tone IDs,
        language IDs, and BERT embeddings suitable for inference.

        Args:
            dtype_override: Optional torch.dtype (not used for integer inputs).

        Returns:
            tuple: (x, x_lengths, sid, tone, language, bert, ja_bert)
        """
        seq_len = 50
        num_symbols = len(self._hps.symbols) if self._hps else 219
        # Phoneme token IDs: [batch, seq_len]
        x = torch.randint(0, num_symbols, (1, seq_len), dtype=torch.long)
        # Sequence lengths: [batch]
        x_lengths = torch.tensor([seq_len], dtype=torch.long)
        # Speaker ID (femaleargonian=0): [batch]
        sid = torch.tensor([0], dtype=torch.long)
        # Tone IDs: [batch, seq_len]
        tone = torch.zeros(1, seq_len, dtype=torch.long)
        # Language IDs: [batch, seq_len]
        language = torch.zeros(1, seq_len, dtype=torch.long)
        # BERT embeddings: [batch, bert_hidden_size, seq_len]
        bert = torch.zeros(1, 1024, seq_len)
        # Japanese BERT embeddings: [batch, ja_bert_hidden_size, seq_len]
        ja_bert = torch.zeros(1, 768, seq_len)
        return x, x_lengths, sid, tone, language, bert, ja_bert
