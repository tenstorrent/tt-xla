# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Phi 4 model loader implementation for causal language modeling
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Phi 4 model variants."""

    PHI_4 = "phi_4"


class ModelLoader(ForgeModel):
    """Phi 4 model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.PHI_4: ModelConfig(
            pretrained_model_name="microsoft/phi-4",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.PHI_4

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="phi-4",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Phi 4 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype.

        Returns:
            torch.nn.Module: The Phi 4 model instance for causal language modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Load pre-trained model from HuggingFace
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Phi 4 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Set up sample messages
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {
                "role": "user",
                "content": "Can you provide ways to eat combinations of bananas and dragonfruits?",
            },
        ]

        # Apply chat template
        result = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Handle both cases: when result is a tensor or a dict
        if torch.is_tensor(result):
            # Result is just input_ids tensor, create dict manually
            inputs = {"input_ids": result, "attention_mask": torch.ones_like(result)}
        else:
            # Result is already a dict
            inputs = result

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    # TODO - Verify this function correct (was AI_GENERATED)
    def decode_output(self, outputs, dtype_override=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass or generated token IDs

        Returns:
            str: Decoded output text
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Check if outputs are token IDs (from generation) or logits
        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            decoded_output = self.tokenizer.decode(outputs)
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            decoded_output = self.tokenizer.decode(next_token_id)

        return decoded_output
