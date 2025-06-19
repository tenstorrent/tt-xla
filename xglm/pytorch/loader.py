# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XGLM model loader implementation
"""

from transformers import AutoTokenizer, XGLMForCausalLM
from ...base import ForgeModel


class ModelLoader(ForgeModel):

    # Shared configuration parameters
    model_name = "facebook/xglm-1.7B"
    text = "My name is Thomas and my main"
    max_length = 256

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load a XGLM model from Hugging Face."""

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

        model = XGLMForCausalLM.from_pretrained(
            cls.model_name, return_dict=False, use_cache=False, **model_kwargs
        )
        model.eval()
        return model

    @classmethod
    def load_inputs(cls):
        """Generate sample inputs for XGLM model."""

        # Ensure tokenizer is initialized
        if not hasattr(cls, "tokenizer"):
            cls.load_model()  # This will initialize the tokenizer

        # Create tokenized inputs
        inputs = cls.tokenizer(
            cls.text,
            max_length=cls.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs
