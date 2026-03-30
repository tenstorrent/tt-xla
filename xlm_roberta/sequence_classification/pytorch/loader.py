# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XLM-RoBERTa model loader implementation for sequence classification.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    """Available XLM-RoBERTa sequence classification model variants."""

    TWITTER_XLM_ROBERTA_BASE_SENTIMENT = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    XLM_ROBERTA_LARGE_HUNGARIAN_BUDGET_CAP_V3 = (
        "poltextlab/xlm-roberta-large-hungarian-budget-cap-v3"
    )
    XLM_ROBERTA_LARGE_HUNGARIAN_LEGISLATIVE_CAP_V3 = (
        "poltextlab/xlm-roberta-large-hungarian-legislative-cap-v3"
    )


class ModelLoader(ForgeModel):
    """XLM-RoBERTa model loader for sequence classification."""

    _VARIANTS = {
        ModelVariant.TWITTER_XLM_ROBERTA_BASE_SENTIMENT: LLMModelConfig(
            pretrained_model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            max_length=128,
        ),
        ModelVariant.XLM_ROBERTA_LARGE_HUNGARIAN_BUDGET_CAP_V3: LLMModelConfig(
            pretrained_model_name="poltextlab/xlm-roberta-large-hungarian-budget-cap-v3",
            max_length=128,
        ),
        ModelVariant.XLM_ROBERTA_LARGE_HUNGARIAN_LEGISLATIVE_CAP_V3: LLMModelConfig(
            pretrained_model_name="poltextlab/xlm-roberta-large-hungarian-legislative-cap-v3",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TWITTER_XLM_ROBERTA_BASE_SENTIMENT

    # NLI sample inputs for MNLI/XNLI variants
    _NLI_PREMISE = (
        "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
    )
    _NLI_HYPOTHESIS = "Angela Merkel ist eine Politikerin."

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.text = _VARIANT_SAMPLE_TEXTS.get(
            self._variant, "Great road trip views! @ Shartlesville, Pennsylvania"
        )

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        group = ModelGroup.VULCAN

        return ModelInfo(
            model="XLM-RoBERTa",
            variant=variant_name,
            group=group,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _is_nli_variant(self):
        """Check if the current variant is an NLI model."""
        return self._variant == ModelVariant.MULTILINGUAL_MINILMV2_L6_MNLI_XNLI

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load XLM-RoBERTa model for sequence classification from Hugging Face."""

        tokenizer_name = self._TOKENIZER_OVERRIDES.get(self._variant, self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

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
        """Prepare sample input for XLM-RoBERTa sequence classification."""
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        if self._is_nli_variant():
            inputs = self.tokenizer(
                self._NLI_PREMISE,
                self._NLI_HYPOTHESIS,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return [inputs["input_ids"], inputs["attention_mask"]]

        inputs = self.tokenizer(
            self.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out, framework_model=None):
        """Decode the model output for sequence classification."""
        predicted_class_id = co_out[0].argmax().item()
        model = framework_model if framework_model is not None else self.model
        if model and hasattr(model, "config") and hasattr(model.config, "id2label"):
            predicted_category = model.config.id2label[predicted_class_id]
            if self._is_nli_variant():
                print(f"Predicted Label: {predicted_category}")
            elif self._variant == ModelVariant.TWITTER_XLM_ROBERTA_BASE_SENTIMENT:
                print(f"Predicted Sentiment: {predicted_category}")
            else:
                print(f"Predicted Category: {predicted_category}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
