# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaMA 3 LLaVA-NeXT 8B model loader implementation for image-text-to-text tasks.
"""

from typing import Optional

from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

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
from ....tools.utils import get_file, cast_input_to_type


class ModelVariant(StrEnum):
    """Available LLaMA 3 LLaVA-NeXT 8B model variants."""

    LLAMA3_LLAVA_NEXT_8B = "Llama3_8B"


class ModelLoader(ForgeModel):
    """LLaMA 3 LLaVA-NeXT 8B model loader for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.LLAMA3_LLAVA_NEXT_8B: ModelConfig(
            pretrained_model_name="llava-hf/llama3-llava-next-8b-hf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA3_LLAVA_NEXT_8B

    sample_text = "What is shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize LLaMA 3 LLaVA-NeXT 8B model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LLaMA 3 LLaVA-NeXT 8B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = LlavaNextProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LLaMA 3 LLaVA-NeXT 8B model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = LlavaNextForConditionalGeneration.from_pretrained(
            str(model_name), **kwargs
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for LLaMA 3 LLaVA-NeXT 8B."""
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
        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        # Preprocess
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        image_sizes = inputs["image_sizes"]

        if dtype_override:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)
            pixel_values = cast_input_to_type(pixel_values, dtype_override)
            image_sizes = cast_input_to_type(image_sizes, dtype_override)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
        }
