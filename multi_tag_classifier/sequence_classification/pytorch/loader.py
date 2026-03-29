# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Multi-tag classifier model loader implementation for sequence classification.
"""
from typing import Optional

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
    """Available multi-tag classifier model variants for sequence classification."""

    CRYPTO_DAPT_V7_RUMOUR_CLEANED = "crypto_dapt_v7_rumour_cleaned"


class ModelLoader(ForgeModel):
    """Multi-tag classifier model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.CRYPTO_DAPT_V7_RUMOUR_CLEANED: ModelConfig(
            pretrained_model_name="HugoGiddins/multi-tag-classifier-full-fine-tune-crypto-dapt-v7-rumour-cleaned",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CRYPTO_DAPT_V7_RUMOUR_CLEANED

    sample_text = "Bitcoin surges past $100k as institutional investors pile in amid ETF approval rumors."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Multi-Tag Classifier",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        inputs = self.tokenizer(
            self.sample_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        import torch

        logits = co_out[0]
        probabilities = torch.sigmoid(logits)
        labels = [
            "FA",
            "FOMO",
            "FUD",
            "Fundraising",
            "Low_Value",
            "Macro",
            "Marketing_Promo",
            "News_Report",
            "OnChain",
            "Project_Update",
            "Rumour",
            "Shilling",
            "TA",
        ]
        predicted = [labels[i] for i, p in enumerate(probabilities[0]) if p > 0.5]
        print(f"Predicted tags: {predicted}")
