# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NanoGPT model loader implementation
"""

from ...base import ForgeModel

from transformers import AutoModel, AutoTokenizer


class ModelLoader(ForgeModel):
    """Loads NanoGPT model and sample input."""

    # Shared configuration parameters
    model_name = "FinancialSupport/NanoGPT"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load pretrained NanoGPT model."""
        model = AutoModel.from_pretrained(
            cls.model_name,
            ignore_mismatched_sizes=True,
            use_cache=False,
            return_dict=False,
        )
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Prepare sample input for NanoGPT model"""

        tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Input prompt
        input_prompt = "The financial market showed signs of volatility"

        # Tokenize input
        inputs = tokenizer(
            input_prompt,
            return_tensors="pt",
            max_length=150,
            padding=True,
            truncation=True,
        )

        return inputs
