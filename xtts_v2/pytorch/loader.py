# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XTTS-v2 model loader implementation for text-to-speech tasks.
"""
import os

import torch
import torch.nn as nn
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


class XttsHifiganWrapper(nn.Module):
    """Wrapper around the XTTS-v2 HiFi-GAN decoder for audio synthesis.

    Exposes the HiFi-GAN vocoder as a clean forward pass that takes
    GPT latents and a speaker embedding to produce audio waveforms.
    """

    def __init__(self, hifigan_decoder):
        super().__init__()
        self.hifigan_decoder = hifigan_decoder

    def forward(self, latents, speaker_embedding):
        return self.hifigan_decoder(latents, g=speaker_embedding)


class ModelVariant(StrEnum):
    """Available XTTS-v2 model variants."""

    XTTS_V2 = "v2"


class ModelLoader(ForgeModel):
    """XTTS-v2 model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.XTTS_V2: ModelConfig(
            pretrained_model_name="coqui/XTTS-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XTTS_V2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._xtts_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="XTTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(
            repo_id=self._variant_config.pretrained_model_name
        )

        config = XttsConfig()
        config.load_json(os.path.join(model_dir, "config.json"))
        xtts_model = Xtts.init_from_config(config)
        xtts_model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)
        self._xtts_model = xtts_model

        model = XttsHifiganWrapper(xtts_model.hifigan_decoder)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # GPT latent output: [batch, sequence_length, decoder_input_dim=1024]
        latents = torch.randn(1, 10, 1024, dtype=dtype)
        # Speaker embedding: [batch, d_vector_dim=512, 1]
        speaker_embedding = torch.randn(1, 512, 1, dtype=dtype)
        return latents, speaker_embedding
