# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kimi-K2-Thinking model loader implementation for causal language modeling.
"""
import os
import sys
from typing import Optional
from unittest.mock import patch

import torch

# Patch missing functions before importing model code that depends on them.
# The model's remote code was written for an older transformers that included
# these helpers; newer versions removed them.
import transformers.utils
import transformers.utils.import_utils

if not hasattr(transformers.utils, "is_flash_attn_greater_or_equal_2_10"):

    def _is_flash_attn_gte_2_10():
        return False

    transformers.utils.is_flash_attn_greater_or_equal_2_10 = _is_flash_attn_gte_2_10
    sys.modules["transformers.utils"].__dict__[
        "is_flash_attn_greater_or_equal_2_10"
    ] = _is_flash_attn_gte_2_10

if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):

    def _is_torch_fx_available():
        return False

    transformers.utils.import_utils.is_torch_fx_available = _is_torch_fx_available
    sys.modules["transformers.utils.import_utils"].__dict__[
        "is_torch_fx_available"
    ] = _is_torch_fx_available

# Patch DynamicCache.from_legacy_cache removed in newer transformers
from transformers.cache_utils import DynamicCache

if not hasattr(DynamicCache, "from_legacy_cache"):

    @classmethod  # type: ignore[misc]
    def _from_legacy_cache(cls, past_key_values=None):
        cache = cls()
        if past_key_values is not None:
            for layer_idx, (key, value) in enumerate(past_key_values):
                cache.update(key, value, layer_idx)
        return cache

    DynamicCache.from_legacy_cache = _from_legacy_cache

if not hasattr(DynamicCache, "to_legacy_cache"):

    def _to_legacy_cache(self):
        legacy_cache = []
        for layer in self.layers:
            legacy_cache.append((layer.keys, layer.values))
        return legacy_cache

    DynamicCache.to_legacy_cache = _to_legacy_cache

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module, get_imports

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


class ModelVariant(StrEnum):
    """Available Kimi-K2-Thinking model variants."""

    KIMI_K2_THINKING = "Kimi-K2-Thinking"


class ModelLoader(ForgeModel):
    """Kimi-K2-Thinking model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.KIMI_K2_THINKING: None,
    }

    DEFAULT_VARIANT = ModelVariant.KIMI_K2_THINKING

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        num_layers: Optional[int] = None,
    ):
        super().__init__(variant)
        self.model_name = "moonshotai/Kimi-K2-Thinking"
        self.tokenizer = None
        self.text = "What is machine learning?"
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Kimi-K2-Thinking",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kimi-K2-Thinking model with reduced config.

        The full model is a 1T-parameter MoE causal language model built on
        the DeepSeek V3 architecture. We load with a reduced configuration
        for testing purposes.

        Args:
            dtype_override: Optional torch.dtype to override the model's dtype.

        Returns:
            torch.nn.Module: The DeepSeek V3 model instance.
        """
        model = None
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)

            if self.num_layers is not None:
                config.num_hidden_layers = self.num_layers
            else:
                config.num_hidden_layers = 2
            config.num_attention_heads = 16
            config.hidden_size = 1024
            config.num_key_value_heads = 16
            config.intermediate_size = 1024 * 4
            config.num_experts_per_tok = 2
            config.q_lora_rank = 256
            config.use_flash_attention = False
            config._attn_implementation = "eager"

            model_kwargs = {
                "attn_implementation": "eager",
                "trust_remote_code": True,
            }
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs

            model_class = get_class_from_dynamic_module(
                "modeling_deepseek.DeepseekV3ForCausalLM",
                self.model_name,
                trust_remote_code=True,
            )
            model = model_class(config)
            model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        return model

    def load_inputs(self, batch_size=1):
        if self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(self.text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
