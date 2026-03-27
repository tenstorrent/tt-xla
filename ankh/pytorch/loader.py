# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ankh protein language model loader implementation for conditional generation on protein sequences.
"""
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
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
    """Available Ankh model variants."""

    ANKH_BASE = "ElnaggarLab/ankh-base"


class ModelLoader(ForgeModel):
    """Ankh protein language model loader for conditional generation on protein sequences."""

    _VARIANTS = {
        ModelVariant.ANKH_BASE: ModelConfig(
            pretrained_model_name="ElnaggarLab/ankh-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ANKH_BASE

    # Short protein sequence for testing
    sample_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAATGFHISDEH"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self._cached_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Ankh",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = T5ForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        # Tokenize protein sequence with spaces between amino acids (ankh convention)
        spaced_sequence = " ".join(list(self.sample_sequence))
        inputs = self.tokenizer(
            spaced_sequence,
            return_tensors="pt",
        )

        # T5-based model requires decoder_input_ids
        decoder_start_token_id = (
            self._cached_model.generation_config.decoder_start_token_id
        )
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
