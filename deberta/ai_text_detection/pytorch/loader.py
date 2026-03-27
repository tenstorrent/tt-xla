# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeBERTa model loader implementation for AI text detection classification.
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
    """Available DeBERTa model variants for AI text detection."""

    DESKLIB_AI_TEXT_DETECTOR_V1_01 = "desklib_ai_text_detector_v1_01"


class ModelLoader(ForgeModel):
    """DeBERTa model loader implementation for AI text detection."""

    _VARIANTS = {
        ModelVariant.DESKLIB_AI_TEXT_DETECTOR_V1_01: ModelConfig(
            pretrained_model_name="desklib/ai-text-detector-v1.01",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DESKLIB_AI_TEXT_DETECTOR_V1_01

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DeBERTa",
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

        sample_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a sample text to test AI text detection capabilities."
        )

        inputs = self.tokenizer(
            sample_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        import torch

        logits = co_out[0]
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class_id = probabilities.argmax(-1).item()
        confidence = probabilities[0][predicted_class_id].item()
        labels = ["Human-written", "AI-generated"]
        print(f"Predicted: {labels[predicted_class_id]} (confidence: {confidence:.4f})")
