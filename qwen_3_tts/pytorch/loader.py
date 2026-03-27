# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS-12Hz-0.6B-Base model loader implementation for text-to-speech tasks.
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
    """Available Qwen3-TTS model variants."""

    TTS_12HZ_0_6B_BASE = "TTS_12Hz_0.6B_Base"


class ModelLoader(ForgeModel):
    """Qwen3-TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.TTS_12HZ_0_6B_BASE: ModelConfig(
            pretrained_model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TTS_12HZ_0_6B_BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen3TTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen3-TTS model."""
        from qwen_tts.core.models import (
            Qwen3TTSConfig,
            Qwen3TTSForConditionalGeneration,
        )
        from transformers import AutoConfig, AutoModel

        pretrained_model_name = self._variant_config.pretrained_model_name

        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

        model = AutoModel.from_pretrained(
            pretrained_model_name,
            torch_dtype=dtype_override if dtype_override is not None else torch.float32,
        )

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Qwen3-TTS model."""
        talker_config = self.model.config.talker_config

        seq_len = 128
        vocab_size = talker_config.vocab_size

        input_ids = torch.randint(0, vocab_size, (1, seq_len))
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
