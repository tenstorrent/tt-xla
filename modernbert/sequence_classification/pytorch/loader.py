# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ModernBERT model loader implementation for sequence classification (propaganda technique detection).
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
    """Available ModernBERT model variants for sequence classification."""

    NCI_TECHNIQUE_CLASSIFIER_V5_2 = "NCI_Technique_Classifier_v5.2"


class ModelLoader(ForgeModel):
    """ModernBERT model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.NCI_TECHNIQUE_CLASSIFIER_V5_2: LLMModelConfig(
            pretrained_model_name="synapti/nci-technique-classifier-v5.2",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NCI_TECHNIQUE_CLASSIFIER_V5_2

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.sample_text = "This is the best product ever made, everyone agrees it is absolutely perfect."

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ModernBERT",
            variant=variant_name,
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

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
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

    def decode_output(self, co_out, framework_model=None):
        predicted_class_id = co_out[0].argmax().item()
        if (
            framework_model
            and hasattr(framework_model, "config")
            and hasattr(framework_model.config, "id2label")
        ):
            predicted_category = framework_model.config.id2label[predicted_class_id]
            print(f"Predicted technique: {predicted_category}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
