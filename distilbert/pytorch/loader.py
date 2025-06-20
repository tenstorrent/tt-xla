# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Distilbert model loader implementation
"""

from transformers import DistilBertForMaskedLM, DistilBertTokenizer
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """Loads Distilbert model and sample input."""

    # Shared configuration parameters
    model_name = "distilbert-base-cased"
    input_prompt = "The capital of France is [MASK]."
    max_length = 128

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load Distilbert model from Hugging Face."""

        # Initialize tokenizer first with default or overridden dtype
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        cls.tokenizer = DistilBertTokenizer.from_pretrained(
            cls.model_name, **tokenizer_kwargs
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = DistilBertForMaskedLM.from_pretrained(cls.model_name, **model_kwargs)
        model.eval()
        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Prepare sample input for Distilbert model"""

        # Data preprocessing
        inputs = cls.tokenizer(
            cls.input_prompt,
            max_length=cls.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs
