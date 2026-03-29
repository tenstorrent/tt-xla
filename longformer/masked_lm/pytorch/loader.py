# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Longformer model loader implementation for masked language modeling.
"""

from transformers import AutoTokenizer, LongformerForMaskedLM
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
    """Available Longformer model variants for masked language modeling."""

    LONGFORMER_BASE_4096 = "Base_4096"
    CLINICAL_LONGFORMER = "Clinical-Longformer"
    SCIBERT_SCIVOCAB_UNCASED_LONG_4096 = "yorko/scibert_scivocab_uncased_long_4096"


class ModelLoader(ForgeModel):
    """Longformer model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.LONGFORMER_BASE_4096: LLMModelConfig(
            pretrained_model_name="allenai/longformer-base-4096",
            max_length=512,
        ),
        ModelVariant.CLINICAL_LONGFORMER: LLMModelConfig(
            pretrained_model_name="yikuan8/Clinical-Longformer",
            max_length=512,
        ),
        ModelVariant.SCIBERT_SCIVOCAB_UNCASED_LONG_4096: LLMModelConfig(
            pretrained_model_name="yorko/scibert_scivocab_uncased_long_4096",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LONGFORMER_BASE_4096

    _SAMPLE_TEXTS = {
        ModelVariant.SCIBERT_SCIVOCAB_UNCASED_LONG_4096: "The [MASK] of neural networks has revolutionized natural language processing.",
    }

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.sample_text = self._SAMPLE_TEXTS.get(
            self._variant, "The capital of France is <mask>."
        )
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant=None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Longformer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LongformerForMaskedLM.from_pretrained(self.model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        inputs = self.load_inputs()
        logits = co_out[0]
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_token = self.tokenizer.decode(predicted_token_id)
        mask_token = self.tokenizer.mask_token
        print(f"The predicted token for the {mask_token} is:", predicted_token)
