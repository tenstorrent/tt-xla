# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LoRA PII Detector BERT model loader implementation for token classification.

This model applies a LoRA adapter (llm-semantic-router/lora_pii_detector_bert-base-uncased_model)
on top of bert-base-uncased for PII (Personally Identifiable Information) detection.
"""

import torch
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
    """Available LoRA PII Detector BERT model variants."""

    LORA_PII_DETECTOR_BERT_BASE_UNCASED = "LoRA_PII_Detector_Bert_Base_Uncased"


class ModelLoader(ForgeModel):
    """LoRA PII Detector BERT model loader for token classification."""

    _VARIANTS = {
        ModelVariant.LORA_PII_DETECTOR_BERT_BASE_UNCASED: ModelConfig(
            pretrained_model_name="llm-semantic-router/lora_pii_detector_bert-base-uncased_model",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LORA_PII_DETECTOR_BERT_BASE_UNCASED

    BASE_MODEL_NAME = "bert-base-uncased"

    sample_text = (
        "John Smith lives at 123 Main Street and his email is john.smith@example.com"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LoRA-PII-Detector-BERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the BERT base model with LoRA PII detector adapter applied.

        Returns:
            torch.nn.Module: The merged LoRA model instance.
        """
        from transformers import BertForTokenClassification, BertTokenizer
        from peft import PeftModel

        self.tokenizer = BertTokenizer.from_pretrained(self.BASE_MODEL_NAME)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        base_model = BertForTokenClassification.from_pretrained(
            self.BASE_MODEL_NAME, **model_kwargs
        )
        model = PeftModel.from_pretrained(
            base_model, self._variant_config.pretrained_model_name
        )
        model = model.merge_and_unload()
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample inputs for PII detection.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            from transformers import BertTokenizer

            self.tokenizer = BertTokenizer.from_pretrained(self.BASE_MODEL_NAME)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for PII token classification.

        Args:
            co_out: Model output
        """
        inputs = self.load_inputs()
        predicted_token_class_ids = co_out[0].argmax(-1)
        predicted_token_class_ids = torch.masked_select(
            predicted_token_class_ids, (inputs["attention_mask"][0] == 1)
        )

        if (
            self.model
            and hasattr(self.model, "config")
            and hasattr(self.model.config, "id2label")
        ):
            predicted_tokens_classes = [
                self.model.config.id2label[t.item()] for t in predicted_token_class_ids
            ]
        else:
            predicted_tokens_classes = [t.item() for t in predicted_token_class_ids]

        print(f"Context: {self.sample_text}")
        print(f"PII Labels: {predicted_tokens_classes}")
