# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InteractiveOmni-4B model loader implementation for multimodal conditional generation.
"""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

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
from ...tools.utils import get_file, cast_input_to_type


class ModelVariant(StrEnum):
    """Available InteractiveOmni model variants."""

    INTERACTIVE_OMNI_4B = "4B"


class ModelLoader(ForgeModel):
    """InteractiveOmni model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.INTERACTIVE_OMNI_4B: ModelConfig(
            pretrained_model_name="sensenova/InteractiveOmni-4B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INTERACTIVE_OMNI_4B

    sample_text = "What is shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize InteractiveOmni model loader."""
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="InteractiveOmni",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the InteractiveOmni model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(str(model_name), **model_kwargs)
        model.eval()

        if self.tokenizer is None:
            self._load_tokenizer()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for InteractiveOmni."""
        if self.tokenizer is None:
            self._load_tokenizer()

        # Load sample image
        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        # Build prompt with image token
        question = f"<image>\n{self.sample_text}"
        messages = [{"role": "user", "content": question}]
        text_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer(
            text_prompt, return_tensors="pt", add_special_tokens=False
        )
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        if dtype_override:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
