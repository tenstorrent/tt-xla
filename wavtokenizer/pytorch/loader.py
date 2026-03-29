# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WavTokenizer model loader for discrete audio codec tokenization.

WavTokenizer is a VQ-VAE style audio codec that compresses raw audio waveforms
into discrete tokens using a single codebook, operating at 24kHz with 75 tokens
per second of audio.

Requires the WavTokenizer repository to be cloned at /tmp/wavtokenizer_repo.
"""

import os
import sys

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

WAVTOKENIZER_REPO_PATH = "/tmp/wavtokenizer_repo"


def _ensure_wavtokenizer_importable():
    """Ensure the WavTokenizer repo is cloned and importable."""
    if not os.path.isdir(WAVTOKENIZER_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/jishengpeng/WavTokenizer.git",
                WAVTOKENIZER_REPO_PATH,
            ]
        )

    if WAVTOKENIZER_REPO_PATH not in sys.path:
        sys.path.insert(0, WAVTOKENIZER_REPO_PATH)


class ModelVariant(StrEnum):
    LARGE_SPEECH_75TOKEN = "Large_Speech_75token"


class ModelLoader(ForgeModel):
    """WavTokenizer model loader for discrete audio codec tokenization.

    Loads the WavTokenizer decoder model that reconstructs audio waveforms
    from discrete codec features, using a custom library and HuggingFace-hosted
    checkpoint.
    """

    _VARIANTS = {
        ModelVariant.LARGE_SPEECH_75TOKEN: ModelConfig(
            pretrained_model_name="novateur/WavTokenizer-large-speech-75token",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_SPEECH_75TOKEN

    # Config YAML is hosted in the base novateur/WavTokenizer repo
    _CONFIG_REPO = "novateur/WavTokenizer"
    _CONFIG_FILENAME = (
        "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    )
    _CKPT_FILENAME = "wavtokenizer_large_speech_320_v2.ckpt"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WavTokenizer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the WavTokenizer decoder model."""
        from huggingface_hub import hf_hub_download

        _ensure_wavtokenizer_importable()
        from decoder.pretrained import WavTokenizer

        # Download config YAML from base WavTokenizer repo
        config_path = hf_hub_download(
            repo_id=self._CONFIG_REPO, filename=self._CONFIG_FILENAME
        )

        # Download checkpoint from the large-speech variant repo
        model_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=self._CKPT_FILENAME,
        )

        model = WavTokenizer.from_pretrained0802(config_path, model_path)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the WavTokenizer model.

        Returns:
            dict: Dictionary with 'wav' tensor (1-second mono audio at 24kHz)
                and 'bandwidth_id' tensor for codec configuration.
        """
        dtype = dtype_override or torch.float32

        # Generate synthetic 1-second mono audio waveform at 24kHz
        torch.manual_seed(42)
        sample_rate = 24000
        wav = torch.randn(1, 1, sample_rate, dtype=dtype)

        bandwidth_id = torch.tensor([0])

        return {"wav": wav, "bandwidth_id": bandwidth_id}
