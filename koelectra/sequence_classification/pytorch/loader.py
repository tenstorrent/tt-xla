# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KoELECTRA model loader implementation for sequence classification.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
    """Available KoELECTRA model variants for sequence classification."""

    CIRCULUS_KOELECTRA_ETHICS_V1 = "circulus_KoELECTRA_Ethics_v1"


class ModelLoader(ForgeModel):
    """KoELECTRA model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.CIRCULUS_KOELECTRA_ETHICS_V1: LLMModelConfig(
            pretrained_model_name="circulus/koelectra-ethics-v1",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CIRCULUS_KOELECTRA_ETHICS_V1

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant=None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="KoELECTRA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model.eval()
        return self.model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        sentence = "이 영화는 정말 재미있었어요"
        inputs = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        import torch

        logits = co_out[0]
        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > 0.5).int()

        for idx, (prob, pred) in enumerate(zip(probabilities[0], predicted_labels[0])):
            label = self.model.config.id2label.get(idx, f"LABEL_{idx}")
            status = "YES" if pred.item() == 1 else "NO"
            print(f"{label}: {status} ({prob.item():.4f})")
