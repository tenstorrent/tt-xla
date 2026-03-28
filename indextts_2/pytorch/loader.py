# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
IndexTTS-2 model loader implementation for text-to-speech tasks.
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


class IndexTTS2GPTWrapper(nn.Module):
    """Wrapper around the IndexTTS-2 GPT (UnifiedVoice) backbone.

    Exposes a clean forward pass that takes pre-computed input embeddings
    and produces mel token logits.
    """

    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt

    def forward(self, inputs_embeds):
        outputs = self.gpt.gpt.transformer(inputs_embeds=inputs_embeds)
        hidden_states = outputs.last_hidden_state
        logits = self.gpt.gpt.lm_head(hidden_states)
        return logits


class ModelVariant(StrEnum):
    """Available IndexTTS-2 model variants."""

    INDEXTTS_2 = "IndexTTS-2"


class ModelLoader(ForgeModel):
    """IndexTTS-2 model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.INDEXTTS_2: ModelConfig(
            pretrained_model_name="IndexTeam/IndexTTS-2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INDEXTTS_2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="IndexTTS-2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from huggingface_hub import snapshot_download
        from indextts.infer_v2 import IndexTTS2

        model_dir = snapshot_download(
            repo_id=self._variant_config.pretrained_model_name,
        )
        tts = IndexTTS2(
            cfg_path=os.path.join(model_dir, "config.yaml"),
            model_dir=model_dir,
            use_fp16=False,
            use_cuda_kernel=False,
            use_deepspeed=False,
        )
        model = IndexTTS2GPTWrapper(tts)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # GPT backbone hidden_size=1280, use a short sequence
        inputs_embeds = torch.randn(1, 32, 1280, dtype=dtype)
        return (inputs_embeds,)
