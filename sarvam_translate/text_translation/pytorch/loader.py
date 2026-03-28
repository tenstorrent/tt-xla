# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sarvam Translate model loader implementation for text translation task.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
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
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Sarvam Translate model variants."""

    SARVAM_TRANSLATE = "Sarvam_Translate"


class ModelLoader(ForgeModel):
    """Sarvam Translate model loader implementation for text translation task."""

    _VARIANTS = {
        ModelVariant.SARVAM_TRANSLATE: LLMModelConfig(
            pretrained_model_name="sarvamai/sarvam-translate",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SARVAM_TRANSLATE

    sample_text = "Hello, how are you today?"
    target_language = "Hindi"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Sarvam Translate",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TRANSLATION,
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
        pretrained_model_name = self._variant_config.pretrained_model_name
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Sarvam Translate model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Sarvam Translate model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)
        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Sarvam Translate model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors that can be fed to the model.
        """
        max_length = self._variant_config.max_length
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)
        messages = [
            {
                "role": "system",
                "content": f"Translate the text below to {self.target_language}.",
            },
            {
                "role": "user",
                "content": self.sample_text,
            },
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(
            [input_text],
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attn_mask = cast_input_to_type(attn_mask, dtype_override)
        return [input_ids, attn_mask]
