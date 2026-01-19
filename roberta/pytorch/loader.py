# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Roberta model implementation for Tenstorrent projects.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

    ROBERTA_BASE_SENTIMENT = "cardiffnlp/twitter-roberta-base-sentiment"


class ModelLoader(ForgeModel):
    """Roberta model loader implementation."""

    _VARIANTS = {
        ModelVariant.ROBERTA_BASE_SENTIMENT: ModelConfig(
            pretrained_model_name="cardiffnlp/twitter-roberta-base-sentiment",
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
        return ModelInfo(
            model="roberta",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.text = """Great road trip views! @ Shartlesville, Pennsylvania"""
        self.max_length = 128
        self.tokenizer = None

    def load_model(self, dtype_override=None):
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

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self):
        """Generate sample inputs for Roberta model."""

        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.load_model()  # This will initialize the tokenizer

        # Create tokenized inputs
        inputs = self.tokenizer.encode(
            self.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass

        Returns:
            str: Decoded answer text
        """
        predicted_value = co_out[0].argmax(-1).item()
        print(f"Predicted Sentiment: {self.model.config.id2label[predicted_value]}")
