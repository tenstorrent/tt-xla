# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WebOrganizer FormatClassifier model loader for web page format classification.
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
    """Available WebOrganizer FormatClassifier model variants."""

    FORMAT_CLASSIFIER = "FormatClassifier"


class ModelLoader(ForgeModel):
    """WebOrganizer FormatClassifier model loader for web page format classification."""

    _VARIANTS = {
        ModelVariant.FORMAT_CLASSIFIER: ModelConfig(
            pretrained_model_name="WebOrganizer/FormatClassifier",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FORMAT_CLASSIFIER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WebOrganizer",
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

        model_kwargs = {
            "trust_remote_code": True,
            "use_memory_efficient_attention": False,
        }
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

        sample_text = (
            "http://www.example.com\n\n"
            "How to Build a REST API with Python and Flask. "
            "In this tutorial, we will walk through the steps to create "
            "a simple REST API using Python and the Flask framework."
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
        logits = co_out[0]
        predicted_class_id = logits.argmax(-1).item()
        label = self.model.config.id2label[predicted_class_id]
        print(f"Predicted Format: {label}")
