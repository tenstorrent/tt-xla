# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeBERTa model loader implementation for masked language modeling.
"""

from transformers import AutoTokenizer, DebertaV2ForMaskedLM
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
    """Available DeBERTa model variants for masked language modeling."""

    DEBERTA_V3_SMALL = "V3_Small"
    DEBERTA_V3_BASE = "V3_Base"
    DEBERTA_V3_LARGE = "V3_Large"
    DEBERTA_V2_TINY_JAPANESE_CHAR_WWM = "V2_Tiny_Japanese_Char_WWM"


class ModelLoader(ForgeModel):
    """DeBERTa model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.DEBERTA_V3_SMALL: LLMModelConfig(
            pretrained_model_name="microsoft/deberta-v3-small",
            max_length=128,
        ),
        ModelVariant.DEBERTA_V3_BASE: LLMModelConfig(
            pretrained_model_name="microsoft/deberta-v3-base",
            max_length=128,
        ),
        ModelVariant.DEBERTA_V3_LARGE: LLMModelConfig(
            pretrained_model_name="microsoft/deberta-v3-large",
            max_length=128,
        ),
        ModelVariant.DEBERTA_V2_TINY_JAPANESE_CHAR_WWM: LLMModelConfig(
            pretrained_model_name="ku-nlp/deberta-v2-tiny-japanese-char-wwm",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEBERTA_V3_BASE

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.sample_text = "The capital of France is [MASK]."
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant=None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DeBERTa",
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

        model = DebertaV2ForMaskedLM.from_pretrained(self.model_name, **model_kwargs)
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
        print("The predicted token for the [MASK] is:", predicted_token)
