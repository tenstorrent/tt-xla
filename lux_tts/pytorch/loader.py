# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LuxTTS model loader implementation for text-to-speech tasks.
"""
import json
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


class LuxTTSWrapper(nn.Module):
    """Wrapper around the ZipVoiceDistill model for speech generation.

    Exposes the flow-matching decoder's sample method as a clean forward pass
    that takes text tokens, prompt tokens, prompt features, and prompt feature
    lengths to produce predicted acoustic features.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        prompt_features,
        prompt_features_lens,
        noise,
        features_lens,
        speech_condition_mask,
    ):
        tokens = [[1, 2, 3, 4, 5]]
        features = torch.cat(
            [prompt_features, noise[:, prompt_features.size(1) :, :]], dim=1
        )
        x_t_end, x_t_end_lens = self.model(
            tokens=tokens,
            features=features,
            features_lens=features_lens,
            noise=noise,
            speech_condition_mask=speech_condition_mask,
            t_start=0.0,
            t_end=1.0,
            num_step=1,
            guidance_scale=torch.tensor([[[3.0]]]),
        )
        return x_t_end


class ModelVariant(StrEnum):
    """Available LuxTTS model variants."""

    LUX_TTS = "LuxTTS"


class ModelLoader(ForgeModel):
    """LuxTTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.LUX_TTS: ModelConfig(
            pretrained_model_name="YatharthS/LuxTTS",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LUX_TTS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model_config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LuxTTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from huggingface_hub import snapshot_download
        from zipvoice.models.zipvoice_distill import ZipVoiceDistill
        from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
        from zipvoice.utils.checkpoint import load_checkpoint

        model_dir = snapshot_download(
            repo_id=self._variant_config.pretrained_model_name
        )

        token_file = os.path.join(model_dir, "tokens.txt")
        model_ckpt = os.path.join(model_dir, "model.pt")
        config_path = os.path.join(model_dir, "config.json")

        with open(config_path, "r") as f:
            self._model_config = json.load(f)

        tokenizer = EmiliaTokenizer(token_file=token_file)
        tokenizer_config = {
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
        }

        zipvoice_model = ZipVoiceDistill(
            **self._model_config["model"],
            **tokenizer_config,
        )
        load_checkpoint(filename=model_ckpt, model=zipvoice_model, strict=True)
        zipvoice_model.eval()

        model = LuxTTSWrapper(zipvoice_model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        feat_dim = self._model_config["model"].get("feat_dim", 100)
        prompt_len = 50
        total_len = 80

        prompt_features = torch.randn(1, prompt_len, feat_dim, dtype=dtype)
        prompt_features_lens = torch.tensor([prompt_len])
        noise = torch.randn(1, total_len, feat_dim, dtype=dtype)
        features_lens = torch.tensor([total_len])
        speech_condition_mask = torch.ones(1, total_len, dtype=torch.bool)
        speech_condition_mask[:, :prompt_len] = False

        return (
            prompt_features,
            prompt_features_lens,
            noise,
            features_lens,
            speech_condition_mask,
        )
