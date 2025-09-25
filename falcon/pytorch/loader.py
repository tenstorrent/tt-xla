# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Falcon model loader implementation for causal language modeling
"""
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """Available Falcon model variants."""

    FALCON_1B = "tiiuae/Falcon3-1B-Base"
    FALCON_3B = "tiiuae/Falcon3-3B-Base"
    FALCON_7B = "tiiuae/Falcon3-7B-Base"
    FALCON_10B = "tiiuae/Falcon3-10B-Base"
    FALCON_MAMBA_7B = "tiiuae/Falcon3-Mamba-7B-Base"
    FALCON_7B_INSTRUCT = "tiiuae/falcon-7b-instruct"


class ModelLoader(ForgeModel):
    """Falcon model loader implementation for causal LM tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.FALCON_1B: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-1B-Base",
        ),
        ModelVariant.FALCON_3B: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-3B-Base",
        ),
        ModelVariant.FALCON_7B: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-7B-Base",
        ),
        ModelVariant.FALCON_10B: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-10B-Base",
        ),
        ModelVariant.FALCON_MAMBA_7B: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-Mamba-7B-Base",
        ),
        ModelVariant.FALCON_7B_INSTRUCT: ModelConfig(
            pretrained_model_name="tiiuae/falcon-7b-instruct",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.FALCON_1B

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        if variant in [
            ModelVariant.FALCON_1B,
            ModelVariant.FALCON_3B,
            ModelVariant.FALCON_7B,
            ModelVariant.FALCON_10B,
        ]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="falcon",
            variant=variant,
            group=group,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        super().__init__(variant)

        # Configuration parameters
        self.input_text_1 = "Write a function to calculate the factorial of a number"
        self.max_length = 512
        self.tokenizer = None
        self.input_text_2 = "Hello, my dog is cute"

    def load_model(self, dtype_override=None):
        """Load and return the Falcon model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Falcon model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Initialize tokenizer
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Falcon model with default settings.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.load_model()  # This will initialize the tokenizer

        if self._variant == ModelVariant.FALCON_7B_INSTRUCT:
            inputs = self.tokenizer(self.input_text_2, return_tensors="pt")
        else:
            inputs = self.tokenizer.encode(
                self.input_text_1,
                add_special_tokens=True,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
            )
        return inputs

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass
            inputs: Optional input tensors used to generate the outputs

        Returns:
            str: Decoded answer text
        """
        if self.tokenizer is None:
            self.load_model()  # This will initialize the tokenizer

        if inputs is None:
            inputs = self.load_inputs()

        response_start = torch.argmax(outputs.start_logits)
        response_end = torch.argmax(outputs.end_logits) + 1
        response_tokens = inputs.input_ids[0, response_start:response_end]

        return self.tokenizer.decode(response_tokens)
