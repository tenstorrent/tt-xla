# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/linear_autoencoder/pytorch_linear_autoencoder.py
"""
Bloom model loader implementation
"""


from ...base import ForgeModel
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelLoader(ForgeModel):
    """Bloom model loader implementation."""

    # Shared configuration parameters
    model_name = "bigscience/bloom-1b1"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the Bloom model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Bloom model instance.
        """
        # Initialize tokenizer first with default or overridden dtype
        tokenizer_kwargs = {"padding_side": "left"}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_name, **tokenizer_kwargs
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForCausalLM.from_pretrained(cls.model_name, **model_kwargs)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Bloom model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """

        # Ensure tokenizer is initialized
        if not hasattr(cls, "tokenizer"):
            cls.load_model(
                dtype_override=dtype_override
            )  # This will initialize the tokenizer

        # Create batch of sample inputs
        cls.test_input = ["This is a sample text from "] * batch_size

        inputs = cls.tokenizer(
            cls.test_input,
            return_tensors="pt",
            max_length=32,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
        )

        return inputs

    @classmethod
    def decode_output(cls, outputs):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass

        Returns:
            str: Decoded answer text
        """
        if not hasattr(cls, "tokenizer"):
            cls.load_model()  # This will initialize the tokenizer

        # Get logits for the last token in each batch
        next_token_logits = outputs.logits[:, -1]
        next_tokens = next_token_logits.softmax(dim=-1).argmax(dim=-1)

        return [cls.tokenizer.decode([token.item()]) for token in next_tokens]
