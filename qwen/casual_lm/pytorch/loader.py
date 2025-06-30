# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen Casual LM model loader implementation
"""


import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM, GenerationConfig
from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ....base import ForgeModel


class ModelLoader(ForgeModel):
    """Qwen Casual LM model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "Qwen/Qwen2.5-1.5B"
        self.tokenizer = None
        self.text = "Hey, are you conscious? Can you talk to me?"

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
            model="Qwen/Qwen2.5-1.5B",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Qwen Casual LM model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Qwen Casual LM model instance.

        """
        # Initialize tokenizer
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, **tokenizer_kwargs
        )

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = Qwen2ForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen Casual LM model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        input_ids = self.tokenizer(self.text, return_tensors="pt").input_ids

        # Use repeat_interleave to expand batch dimension
        input_ids = input_ids.repeat_interleave(batch_size, dim=0)

        generation_config = GenerationConfig(max_length=30)
        arguments = {"input_ids": input_ids, "generation_config": generation_config}

        return arguments

    def decode_output(self, output, dtype_override=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass
            inputs: Optional input tensors used to generate the outputs
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            str: Decoded answer text
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        logits = output.logits if hasattr(output, "logits") else output[0]
        token_ids = torch.argmax(logits, dim=-1)
        gen_text = self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return gen_text
