# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nucleotide Transformer v2 model loader implementation for masked language modeling.
"""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from typing import Optional

from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available Nucleotide Transformer model variants for masked language modeling."""

    V2_50M_MULTI_SPECIES = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"


class ModelLoader(ForgeModel):
    """Nucleotide Transformer v2 model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.V2_50M_MULTI_SPECIES: LLMModelConfig(
            pretrained_model_name="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2_50M_MULTI_SPECIES

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Nucleotide Transformer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        # Sample DNA sequence
        dna_sequence = "ATTCCGATTCCGATTCCG"

        max_length = self._variant_config.max_length
        tokens_ids = self.tokenizer(
            dna_sequence,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return tokens_ids

    def decode_output(self, outputs, inputs=None):
        if inputs is None:
            inputs = self.load_inputs()

        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs

        # Get predicted tokens
        predicted_token_ids = torch.argmax(logits, dim=-1)
        predicted_tokens = self.tokenizer.decode(predicted_token_ids[0])
        return predicted_tokens

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.config
