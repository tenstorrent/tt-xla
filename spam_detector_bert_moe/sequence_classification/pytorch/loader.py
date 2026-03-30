# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Spam Detector BERT MoE model loader implementation for sequence classification (spam detection).
"""
from typing import Optional

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available Spam Detector BERT MoE model variants for sequence classification."""

    SPAM_DETECTOR_BERT_MOE_V2_2 = "Spam_Detector_BERT_MoE_v2_2"


class ModelLoader(ForgeModel):
    """Spam Detector BERT MoE model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.SPAM_DETECTOR_BERT_MOE_V2_2: LLMModelConfig(
            pretrained_model_name="AntiSpamInstitute/spam-detector-bert-MoE-v2.2",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SPAM_DETECTOR_BERT_MOE_V2_2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Spam Detector BERT MoE",
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

        sample_text = "Congratulations! You've won a free iPhone. Click here to claim your prize now!"

        inputs = self.tokenizer(
            sample_text,
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out, framework_model=None):
        predicted_class_id = co_out[0].argmax().item()
        if (
            framework_model
            and hasattr(framework_model, "config")
            and hasattr(framework_model.config, "id2label")
        ):
            predicted_label = framework_model.config.id2label[predicted_class_id]
            print(f"Predicted label: {predicted_label}")
        else:
            labels = {0: "Not Spam", 1: "Spam"}
            print(
                f"Predicted label: {labels.get(predicted_class_id, predicted_class_id)}"
            )
