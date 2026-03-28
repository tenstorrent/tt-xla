# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Phi 3.5 model loader implementation for causal language modeling
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Phi 3.5 model variants."""

    MINI_INSTRUCT = "Mini_Instruct"
    MOE_INSTRUCT = "Phi_3.5_Moe_Instruct"
    TINY_RANDOM_MOE = "Phi_3.5_Moe_Tiny_Random"


class ModelLoader(ForgeModel):
    """Phi 3.5 model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.MINI_INSTRUCT: ModelConfig(
            pretrained_model_name="microsoft/Phi-3.5-mini-instruct",
        ),
        ModelVariant.MOE_INSTRUCT: ModelConfig(
            pretrained_model_name="microsoft/Phi-3.5-MoE-instruct",
        ),
        ModelVariant.TINY_RANDOM_MOE: ModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/phi-3.5-moe-tiny-random",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MINI_INSTRUCT

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
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        group = ModelGroup.RED
        if variant == ModelVariant.TINY_RANDOM_MOE:
            group = ModelGroup.VULCAN

        return ModelInfo(
            model="Phi-3",
            variant=variant,
            group=group,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        if self._variant == ModelVariant.TINY_RANDOM_MOE:
            tokenizer_kwargs["trust_remote_code"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            **tokenizer_kwargs,
        )
        # Ensure padding token is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Phi 3.5 model instance for this instance's variant."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)
        model_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model_kwargs = {
            "use_cache": False,
            "torch_dtype": model_dtype,
        }
        if self._variant == ModelVariant.TINY_RANDOM_MOE:
            model_kwargs["trust_remote_code"] = True
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Phi 3.5 model with this instance's variant settings."""
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)
        prompt = [
            {
                "role": "user",
                "content": "Can you provide ways to eat combinations of bananas and dragonfruits?",
            },
        ]
        chat_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self._variant != ModelVariant.TINY_RANDOM_MOE:
            chat_kwargs["enable_thinking"] = True
        text = self.tokenizer.apply_chat_template(prompt, **chat_kwargs)
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs
