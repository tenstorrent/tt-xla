# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Svara TTS v1 model loader implementation for text-to-speech tasks.

Svara TTS is an Orpheus-style multilingual TTS model built on LlamaForCausalLM
that generates discrete audio tokens for 19 languages (18 Indic + Indian English).
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
    """Available Svara TTS model variants."""

    SVARA_TTS_V1 = "svara_tts_v1"


class ModelLoader(ForgeModel):
    """Svara TTS v1 model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.SVARA_TTS_V1: ModelConfig(
            pretrained_model_name="kenpath/svara-tts-v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SVARA_TTS_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="SvaraTTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForCausalLM

        pretrained_model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override or torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, *, dtype_override=None, **kwargs):
        from transformers import AutoTokenizer

        pretrained_model_name = self._variant_config.pretrained_model_name
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        # Sample Hindi text with Orpheus-style emotion tag
        text = "नमस्ते, मैं स्वरा हूं। <happy>"
        tokens = tokenizer(text, return_tensors="pt")

        return tokens
