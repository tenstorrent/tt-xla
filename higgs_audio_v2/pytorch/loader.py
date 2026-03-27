# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Higgs Audio V2 Generation 3B Base model loader implementation for text-to-speech tasks
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
    """Available Higgs Audio V2 model variants."""

    GENERATION_3B_BASE = "Generation_3B_Base"


class ModelLoader(ForgeModel):
    """Higgs Audio V2 model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.GENERATION_3B_BASE: ModelConfig(
            pretrained_model_name="eustlb/higgs-audio-v2-generation-3B-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GENERATION_3B_BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="HiggsAudioV2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Higgs Audio V2 model."""
        from transformers import HiggsAudioV2Model

        pretrained_model_name = self._variant_config.pretrained_model_name

        model = HiggsAudioV2Model.from_pretrained(
            pretrained_model_name,
            torch_dtype=dtype_override if dtype_override is not None else torch.float32,
        )

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Higgs Audio V2 model."""
        config = self.model.config

        seq_len = 128
        num_codebooks = config.num_codebooks
        num_audio_frames = 64

        # Text input tokens
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)

        # Audio codebook tokens: (batch, num_audio_frames, num_codebooks)
        audio_input_ids = torch.randint(
            0, config.codebook_size, (1, num_audio_frames, num_codebooks)
        )
        audio_input_ids_mask = torch.ones(1, num_audio_frames, dtype=torch.bool)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_input_ids": audio_input_ids,
            "audio_input_ids_mask": audio_input_ids_mask,
        }
