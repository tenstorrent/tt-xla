# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DNABERT model loader implementation for masked language modeling on DNA sequences.
"""

from transformers import AutoTokenizer, AutoModelForMaskedLM
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
    """Available DNABERT model variants for masked language modeling."""

    DNA_BERT_6 = "zhihan1996/DNA_bert_6"


class ModelLoader(ForgeModel):
    """DNABERT model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.DNA_BERT_6: LLMModelConfig(
            pretrained_model_name="zhihan1996/DNA_bert_6",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DNA_BERT_6

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DNABERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        # DNABERT uses k-mer tokenization (6-mers for DNA_bert_6).
        # Input DNA sequences must be pre-tokenized into space-separated k-mers
        # with one token replaced by [MASK].
        dna_sequence = "ACT GAC TGA CTG ACT GAC TGA CTG [MASK] CTG ACT GAC"

        max_length = self._variant_config.max_length
        inputs = self.tokenizer(
            dna_sequence,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_token = self.tokenizer.decode(predicted_token_id)
        return predicted_token
