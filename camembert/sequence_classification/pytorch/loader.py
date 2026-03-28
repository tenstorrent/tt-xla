# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CamemBERT model loader implementation for sequence classification.
CamemBERT is a RoBERTa-based French language model used here for toxicity classification.
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
    """Available CamemBERT sequence classification model variants."""

    EISTAKOVSKII_FRENCH_TOXICITY_CLASSIFIER_PLUS_V2 = (
        "EIStakovskii_french_toxicity_classifier_plus_v2"
    )


class ModelLoader(ForgeModel):
    """CamemBERT model loader for sequence classification."""

    _VARIANTS = {
        ModelVariant.EISTAKOVSKII_FRENCH_TOXICITY_CLASSIFIER_PLUS_V2: LLMModelConfig(
            pretrained_model_name="EIStakovskii/french_toxicity_classifier_plus_v2",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EISTAKOVSKII_FRENCH_TOXICITY_CLASSIFIER_PLUS_V2

    _SAMPLE_TEXTS = {
        ModelVariant.EISTAKOVSKII_FRENCH_TOXICITY_CLASSIFIER_PLUS_V2: "J'aime ta coiffure, elle est très jolie.",
    }

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.text = self._SAMPLE_TEXTS.get(
            self._variant, "J'aime ta coiffure, elle est très jolie."
        )

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="CamemBERT",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load CamemBERT model for sequence classification from Hugging Face."""

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for CamemBERT sequence classification."""
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for sequence classification."""
        predicted_value = co_out[0].argmax(-1).item()

        print(f"Predicted Category: {self.model.config.id2label[predicted_value]}")
