# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tarsier model loader implementation for video/image description generation.
"""

from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor

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
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Tarsier model variants."""

    TARSIER_7B = "7B"


class ModelLoader(ForgeModel):
    """Tarsier model loader for video/image description generation."""

    _VARIANTS = {
        ModelVariant.TARSIER_7B: ModelConfig(
            pretrained_model_name="omni-research/Tarsier-7b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TARSIER_7B

    sample_text = "Describe this image in detail."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Tarsier",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Tarsier model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for Tarsier."""
        if self.processor is None:
            self._load_processor()

        # Load a sample image
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]

        if dtype_override:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)
            pixel_values = cast_input_to_type(pixel_values, dtype_override)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
