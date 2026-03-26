# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kimi K2.5 model loader implementation.
"""
import torch
import os
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.dynamic_module_utils import get_imports
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


class ModelVariant(StrEnum):
    """Available Kimi K2.5 model variants."""

    KIMI_K2_5 = "Kimi-K2.5"


class ModelLoader(ForgeModel):
    """Kimi K2.5 model loader implementation."""

    _VARIANTS = {
        ModelVariant.KIMI_K2_5: None,
    }

    DEFAULT_VARIANT = ModelVariant.KIMI_K2_5

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses a reduced default.
        """
        super().__init__(variant)
        self.model_name = "moonshotai/Kimi-K2.5"
        self.tokenizer = None
        self.text = "What is machine learning?"
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Kimi-K2.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kimi K2.5 model instance with reduced config.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Kimi K2.5 model instance.
        """
        model = None
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)

            # Reduce config for testing - the full model is 1T parameters
            text_config = config.text_config
            if self.num_layers is not None:
                text_config.num_hidden_layers = self.num_layers
            else:
                text_config.num_hidden_layers = 2
            text_config.num_attention_heads = 16
            text_config.hidden_size = 1024
            text_config.num_key_value_heads = 16
            text_config.intermediate_size = 1024 * 4
            text_config.num_experts_per_tok = 2
            text_config.q_lora_rank = 256
            text_config.use_flash_attention = False

            model_kwargs = {"attn_implementation": "eager", "trust_remote_code": True}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs

            model = AutoModelForCausalLM.from_config(config, **model_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        return model

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the Kimi K2.5 model.

        Args:
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(self.text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
