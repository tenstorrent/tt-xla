# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Lingo-Judge model loader implementation for text classification.
"""
from typing import Optional

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available Lingo-Judge model variants for text classification."""

    LINGO_JUDGE = "Lingo_Judge"


class ModelLoader(ForgeModel):
    """Lingo-Judge model loader implementation for text classification."""

    _VARIANTS = {
        ModelVariant.LINGO_JUDGE: ModelConfig(
            pretrained_model_name="wayveai/Lingo-Judge",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LINGO_JUDGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Lingo-Judge",
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
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        question = "What is the color of the traffic light?"
        reference_answer = "The traffic light is red."
        predicted_answer = "The traffic light shows a red signal."

        text = (
            f"[CLS] Question: {question} "
            f"Answer: {reference_answer} "
            f"Student: {predicted_answer}"
        )

        inputs = self.tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        import torch

        logits = co_out[0]
        score = torch.sigmoid(logits).item()
        prediction = "correct" if score > 0.5 else "incorrect"
        print(f"Score: {score:.4f}, Prediction: {prediction}")
