# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Parler-TTS model loader implementation for text-to-speech tasks.
"""
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


class ParlerTTSDecoderWrapper(nn.Module):
    """Wrapper around ParlerTTSForConditionalGeneration decoder.

    Exposes the decoder forward pass that takes encoder hidden states
    and decoder input IDs to produce codec logits for speech synthesis.
    """

    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
        self.lm_heads = model.lm_heads

    def forward(self, encoder_hidden_states, decoder_input_ids):
        outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        lm_logits = torch.stack(
            [head(outputs.last_hidden_state) for head in self.lm_heads], dim=1
        )
        return lm_logits


class ModelVariant(StrEnum):
    """Available Parler-TTS model variants."""

    MINI_V1_1 = "mini-v1.1"


class ModelLoader(ForgeModel):
    """Parler-TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.MINI_V1_1: ModelConfig(
            pretrained_model_name="parler-tts/parler-tts-mini-v1.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINI_V1_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Parler-TTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from parler_tts import ParlerTTSForConditionalGeneration

        self._model = ParlerTTSForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype_override or torch.float32,
        )
        model = ParlerTTSDecoderWrapper(self._model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        from transformers import AutoTokenizer

        dtype = dtype_override or torch.float32

        description_tokenizer = AutoTokenizer.from_pretrained(
            self._model.config.text_encoder._name_or_path
        )
        description = "A female speaker with a clear voice."
        input_ids = description_tokenizer(description, return_tensors="pt").input_ids

        encoder_outputs = self._model.text_encoder(input_ids=input_ids)
        encoder_hidden_states = encoder_outputs.last_hidden_state.to(dtype)

        # Single-step decoder input: BOS token for each codebook
        num_codebooks = self._model.config.decoder.num_codebooks
        decoder_input_ids = (
            torch.ones(num_codebooks, 1, dtype=torch.long)
            * self._model.config.decoder.bos_token_id
        )

        return encoder_hidden_states, decoder_input_ids
