# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-OSS Safeguard model loader implementation for causal language modeling tasks.
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import Mxfp4Config
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available GPT-OSS Safeguard model variants."""

    GPT_OSS_SAFEGUARD_20B = "20B"


class ModelLoader(ForgeModel):
    """GPT-OSS Safeguard model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.GPT_OSS_SAFEGUARD_20B: LLMModelConfig(
            pretrained_model_name="openai/gpt-oss-safeguard-20b",
            max_length=256,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GPT_OSS_SAFEGUARD_20B

    # Sample messages for inference
    messages = [
        {"role": "user", "content": "Is this content safe?"},
    ]

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
        self.config = None
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
        return ModelInfo(
            model="GPT-OSS-Safeguard",
            variant=variant,
            group=ModelGroup.VULCAN,
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
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GPT-OSS Safeguard model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use bfloat16 as default.

        Returns:
            torch.nn.Module: The GPT-OSS Safeguard model instance for causal language modeling.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        quantization_config = Mxfp4Config(dequantize=True)
        self.load_config()

        model_kwargs = {
            "config": self.config,
            "quantization_config": quantization_config,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        self.model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the GPT-OSS Safeguard model.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
        )
        if (
            hasattr(self.model.config, "sliding_window")
            and self.model.config.sliding_window is not None
        ):
            self.model.config.sliding_window = inputs["input_ids"].shape[1]

        return inputs

    def load_config(self):
        """Load and return the configuration for the GPT-OSS Safeguard model.

        Returns:
            The configuration object for the model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        if self.num_layers is not None:
            self.config.num_hidden_layers = self.num_layers

        return self.config
