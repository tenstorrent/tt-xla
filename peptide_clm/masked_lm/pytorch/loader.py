# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PeptideCLM model loader implementation for masked language modeling on peptide sequences.

This model uses a custom SMILES tokenizer not available via the transformers
library, so we generate synthetic inputs directly from the model config.
"""

import torch
from typing import Optional

from transformers import AutoModelForMaskedLM, AutoConfig

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available PeptideCLM model variants."""

    PEPTIDE_CLM_23M_ALL = "aaronfeller/PeptideCLM-23M-all"


class ModelLoader(ForgeModel):
    """PeptideCLM model loader implementation for masked language modeling on peptide sequences."""

    _VARIANTS = {
        ModelVariant.PEPTIDE_CLM_23M_ALL: ModelConfig(
            pretrained_model_name="aaronfeller/PeptideCLM-23M-all",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PEPTIDE_CLM_23M_ALL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PeptideCLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.config is None:
            self._load_config()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        if self.config is None:
            self._load_config()

        seq_length = 128
        vocab_size = self.config.vocab_size

        input_ids = torch.randint(0, vocab_size, (1, seq_length))
        attention_mask = torch.ones(1, seq_length, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode_output(self, outputs):
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        predicted_token_ids = logits[0].argmax(axis=-1)

        return f"Output token IDs: {predicted_token_ids.tolist()}"
