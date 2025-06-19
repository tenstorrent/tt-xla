# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RoBERTa model loader implementation
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ...base import ForgeModel


class ModelLoader(ForgeModel):

    # Shared configuration parameters
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    text = """Great road trip views! @ Shartlesville, Pennsylvania"""
    max_length = 128

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load a RoBERTa model from Hugging Face."""

        # Initialize tokenizer first with default or overridden dtype
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_name, **tokenizer_kwargs
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForSequenceClassification.from_pretrained(
            cls.model_name, return_dict=False, **model_kwargs
        )
        model.eval()
        return model

    @classmethod
    def load_inputs(cls):
        """Generate sample inputs for RoBERTa model."""

        # Ensure tokenizer is initialized
        if not hasattr(cls, "tokenizer"):
            cls.load_model()  # This will initialize the tokenizer

        # Create tokenized inputs
        inputs = cls.tokenizer.encode(
            cls.text,
            max_length=cls.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs
