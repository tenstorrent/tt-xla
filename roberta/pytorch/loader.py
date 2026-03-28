# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Roberta model implementation for Tenstorrent projects.
"""

from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available Roberta model variants."""

    ROBERTA_BASE_SENTIMENT = "Base_Sentiment"
    ROBERTA_BASE_SENTIMENT_LATEST = "Base_Sentiment_Latest"
    ROBERTA_BASE_OFFENSIVE = "Base_Offensive"
    ROBERTA_LARGE_MNLI = "Large_MNLI"
    MANHTEKY123_COMMENT_CLASSIFICATION = "manhteky123_Comment_Classification"


class ModelLoader(ForgeModel):
    """Roberta model loader implementation."""

    _VARIANTS = {
        ModelVariant.ROBERTA_BASE_SENTIMENT: ModelConfig(
            pretrained_model_name="cardiffnlp/twitter-roberta-base-sentiment",
        ),
        ModelVariant.ROBERTA_BASE_SENTIMENT_LATEST: ModelConfig(
            pretrained_model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        ),
        ModelVariant.ROBERTA_BASE_OFFENSIVE: ModelConfig(
            pretrained_model_name="cardiffnlp/twitter-roberta-base-offensive",
        ),
        ModelVariant.ROBERTA_LARGE_MNLI: ModelConfig(
            pretrained_model_name="FacebookAI/roberta-large-mnli",
        ),
        ModelVariant.MANHTEKY123_COMMENT_CLASSIFICATION: ModelConfig(
            pretrained_model_name="manhteky123/comment-classification",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.ROBERTA_BASE_SENTIMENT

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

        group = ModelGroup.GENERALITY
        if variant_name in (
            ModelVariant.ROBERTA_BASE_SENTIMENT_LATEST,
            ModelVariant.ROBERTA_BASE_OFFENSIVE,
            ModelVariant.ROBERTA_LARGE_MNLI,
            ModelVariant.MANHTEKY123_COMMENT_CLASSIFICATION,
        ):
            group = ModelGroup.VULCAN

        return ModelInfo(
            model="RoBERTa",
            variant=variant_name,
            group=group,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    # NLI sample inputs for NLI-based variants
    _MNLI_PREMISE = (
        "Calcutta seems to be the only other production center having any "
        "pretensions to artistic creativity at all, but ironically you're "
        "actually more likely to see the works of Satyajit Ray or Mrinal Sen "
        "shown in Europe or North America than in India itself."
    )
    _NLI_HYPOTHESIS = "Most of Mrinal Sen's work can be found in European collections."

    _SAMPLE_TEXTS = {
        ModelVariant.GARAK_ROBERTA_TOXICITY: "This is a perfectly normal and friendly comment.",
    }

    _SAMPLE_TEXTS = {
        ModelVariant.ROBERTA_BASE_SUICIDE_PREDICTION: "I like you. I love you",
    }

    # Chinese sample text for Dianping variant
    _DIANPING_TEXT = "这家餐厅的食物非常好吃，服务也很周到，下次还会再来。"

    def __init__(self, variant=None, num_layers: Optional[int] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)

        # Configuration parameters
        self.text = """Great road trip views! @ Shartlesville, Pennsylvania"""
        if self._variant == ModelVariant.ROBERTA_BASE_DIANPING_CHINESE:
            self.text = self._DIANPING_TEXT
        self.max_length = 128
        self.tokenizer = None
        self.num_layers = num_layers

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load a Roberta model from Hugging Face."""

        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Initialize tokenizer first with default or overridden dtype
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def _is_mnli_variant(self):
        """Check if the current variant is an MNLI model."""
        return self._variant in (
            ModelVariant.ROBERTA_BASE_MNLI,
            ModelVariant.ROBERTA_LARGE_MNLI,
        )

    def load_inputs(self):
        """Generate sample inputs for Roberta model."""

        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.load_model()  # This will initialize the tokenizer

        if self._is_nli_variant():
            # NLI-based variants use premise/hypothesis pairs
            inputs = self.tokenizer(
                self._NLI_PREMISE,
                self._NLI_HYPOTHESIS,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return [inputs["input_ids"], inputs["attention_mask"]]

        if self._is_spam_variant():
            text = "Congratulations! You've won a free ticket. Call now to claim your prize!"
        else:
            text = self.text

        # Create tokenized inputs for single-text classification
        text = self._SAMPLE_TEXTS.get(self._variant, self.text)
        inputs = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Helper method to decode model outputs into human-readable text.

        Args:
            co_out: Model output from a forward pass

        Returns:
            str: Decoded answer text
        """
        if self._is_multi_label_variant():
            probs = torch.sigmoid(co_out[0])
            threshold = 0.5
            predicted_indices = (probs > threshold).nonzero(as_tuple=True)[1].tolist()
            labels = [self.model.config.id2label[idx] for idx in predicted_indices]
            print(f"Predicted Topics: {labels}")
        elif self._is_mnli_variant():
            predicted_value = co_out[0].argmax(-1).item()
            label = self.model.config.id2label[predicted_value]
            print(f"Predicted Label: {label}")
        elif self._variant == ModelVariant.ROBERTA_BASE_OFFENSIVE:
            print(f"Predicted Offensiveness: {label}")
        else:
            predicted_value = co_out[0].argmax(-1).item()
            label = self.model.config.id2label[predicted_value]
            print(f"Predicted Sentiment: {label}")
