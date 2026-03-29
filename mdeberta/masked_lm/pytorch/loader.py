# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mDeBERTa V3 model loader implementation for masked language modeling.

Uses the DebertaV2Model base encoder from mDeBERTa V3 checkpoints. Supports:
- microsoft/mdeberta-v3-base: multilingual DeBERTa V3 pre-trained on CC100 (100+ languages)
- lighthouse/mdeberta-v3-base-kor-further: further pre-trained on ~40GB Korean text

The checkpoint's LM head weights use non-standard naming that is incompatible with
transformers' DebertaV2ForMaskedLM, so we load the base encoder only.
"""

from transformers import AutoTokenizer, DebertaV2Model
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
    """Available mDeBERTa V3 model variants."""

    MDEBERTA_V3_BASE = "mDeBERTa_V3_Base"
    MDEBERTA_V3_BASE_KOR_FURTHER = "mDeBERTa_V3_Base_Kor_Further"


class ModelLoader(ForgeModel):
    """mDeBERTa V3 model loader implementation."""

    _VARIANTS = {
        ModelVariant.MDEBERTA_V3_BASE: LLMModelConfig(
            pretrained_model_name="microsoft/mdeberta-v3-base",
            max_length=128,
        ),
        ModelVariant.MDEBERTA_V3_BASE_KOR_FURTHER: LLMModelConfig(
            pretrained_model_name="lighthouse/mdeberta-v3-base-kor-further",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MDEBERTA_V3_BASE

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.sample_text = "The capital of France is Paris."
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant=None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="mDeBERTa",
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

        model = DebertaV2Model.from_pretrained(self.model_name, **model_kwargs)
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
