# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cross-Encoder model loader implementation for natural language inference.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
    """Available Cross-Encoder model variants for NLI."""

    NLI_DEBERTA_V3_LARGE = "nli-deberta-v3-large"


class ModelLoader(ForgeModel):
    """Cross-Encoder model loader implementation for natural language inference."""

    _VARIANTS = {
        ModelVariant.NLI_DEBERTA_V3_LARGE: ModelConfig(
            pretrained_model_name="cross-encoder/nli-deberta-v3-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NLI_DEBERTA_V3_LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="CrossEncoder",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
            )

        premise = "A man is eating food."
        hypothesis = "A man is eating a meal."

        inputs = self.tokenizer(
            premise,
            hypothesis,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float32:
                        inputs[key] = value.to(dtype_override)

        return inputs

    def decode_output(self, co_out):
        logits = co_out[0]
        predicted_class_id = logits.argmax(-1).item()
        labels = ["contradiction", "entailment", "neutral"]
        print(f"Predicted: {labels[predicted_class_id]}")
