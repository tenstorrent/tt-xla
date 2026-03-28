# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RuBERT model loader implementation for sequence classification (sentiment analysis).
"""

from transformers import AutoModelForSequenceClassification, BertTokenizerFast
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
    """Available RuBERT model variants for sequence classification."""

    RUBERT_BASE_CASED_SENTIMENT_RUSENTIMENT = "Base_Cased_Sentiment_RuSentiment"
    R1CHAR9_RUBERT_BASE_CASED_RUSSIAN_SENTIMENT = (
        "r1char9_RuBERT_Base_Cased_Russian_Sentiment"
    )


class ModelLoader(ForgeModel):
    """RuBERT model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.RUBERT_BASE_CASED_SENTIMENT_RUSENTIMENT: LLMModelConfig(
            pretrained_model_name="blanchefort/rubert-base-cased-sentiment-rusentiment",
            max_length=128,
        ),
        ModelVariant.R1CHAR9_RUBERT_BASE_CASED_RUSSIAN_SENTIMENT: LLMModelConfig(
            pretrained_model_name="r1char9/rubert-base-cased-russian-sentiment",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RUBERT_BASE_CASED_SENTIMENT_RUSENTIMENT

    _SAMPLE_TEXTS = {
        ModelVariant.RUBERT_BASE_CASED_SENTIMENT_RUSENTIMENT: "Мне очень понравился этот фильм, он был замечательным!",
        ModelVariant.R1CHAR9_RUBERT_BASE_CASED_RUSSIAN_SENTIMENT: "Привет, ты мне нравишься!",
    }

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.text = self._SAMPLE_TEXTS.get(
            self._variant,
            "Мне очень понравился этот фильм, он был замечательным!",
        )
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="RuBERT",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load RuBERT model for sequence classification from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The RuBERT model instance.
        """
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for RuBERT sequence classification.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
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
        """Decode the model output for sequence classification.

        Args:
            co_out: Model output
        """
        predicted_value = co_out[0].argmax(-1).item()
        print(f"Predicted Sentiment: {self.model.config.id2label[predicted_value]}")
