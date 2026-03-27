# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepCogito model loader implementation for text generation tasks
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    """Available DeepCogito model variants."""

    V1_PREVIEW_LLAMA_3B = "v1_Preview_Llama_3B"
    V1_PREVIEW_QWEN_32B = "v1_Preview_Qwen_32B"


class ModelLoader(ForgeModel):
    """DeepCogito model loader implementation for text generation tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.V1_PREVIEW_LLAMA_3B: ModelConfig(
            pretrained_model_name="deepcogito/cogito-v1-preview-llama-3B",
        ),
        ModelVariant.V1_PREVIEW_QWEN_32B: ModelConfig(
            pretrained_model_name="deepcogito/cogito-v1-preview-qwen-32B",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.V1_PREVIEW_LLAMA_3B

    # Shared configuration parameters
    prompt = "Give me a short introduction to LLMs."
    system_message = "You are a pirate chatbot who always responds in pirate speak!"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        group = ModelGroup.GENERALITY
        if variant == ModelVariant.V1_PREVIEW_QWEN_32B:
            group = ModelGroup.VULCAN

        return ModelInfo(
            model="DeepCogito",
            variant=variant,
            group=group,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DeepCogito model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use float32.

        Returns:
            torch.nn.Module: The DeepCogito model instance for text generation.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load pre-trained model from HuggingFace
        model_kwargs = {"torch_dtype": torch.float32, "device_map": None}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the DeepCogito model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create chat template
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )

        # Tokenize the input
        inputs = self.tokenizer([text], return_tensors="pt", truncation=True)

        return inputs
