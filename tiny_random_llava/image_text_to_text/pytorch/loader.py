# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tiny Random LLaVA model loader implementation for image-text-to-text tasks.
"""

from typing import Optional

from datasets import load_dataset
from transformers import LlavaForConditionalGeneration, AutoProcessor

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
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Tiny Random LLaVA model variants."""

    TINY_RANDOM_LLAVA = "tiny_random_llava"


class ModelLoader(ForgeModel):
    """Tiny Random LLaVA model loader for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM_LLAVA: ModelConfig(
            pretrained_model_name="katuni4ka/tiny-random-llava",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM_LLAVA

    sample_text = "What is shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Tiny Random LLaVA model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="TinyRandomLLaVA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Tiny Random LLaVA model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = LlavaForConditionalGeneration.from_pretrained(str(model_name), **kwargs)
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for Tiny Random LLaVA."""
        if self.processor is None:
            self._load_processor()

        # Build prompt using chat template
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
            conversation, padding=True, add_generation_prompt=True
        )

        # Load sample image
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        # Preprocess
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
