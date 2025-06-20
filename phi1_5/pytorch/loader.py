# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Phi1_5 model loader implementation
"""

from transformers import PhiForTokenClassification, AutoTokenizer
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """Loads Phi1_5 model and sample input."""

    # Shared configuration parameters
    model_name = "microsoft/phi-1_5"
    input_prompt = "HuggingFace is a company based in Paris and New York"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load Phi1_5 model from Hugging Face."""

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

        model = PhiForTokenClassification.from_pretrained(
            cls.model_name, return_dict=False, use_cache=False, **model_kwargs
        )
        model.eval()
        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Prepare sample input for Phi1_5 model"""

        # input_prompt
        inputs = cls.tokenizer(cls.input_prompt, return_tensors="pt")

        return inputs
