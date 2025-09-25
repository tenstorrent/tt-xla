# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaVA model loader implementation for multimodal conditional generation.
"""

import os
import re
from typing import Optional

import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

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
from ...tools.utils import get_file
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available LLaVA model variants."""

    LLAVA_1_5_7B = "1_5_7b"


class ModelLoader(ForgeModel):
    """LLaVA model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.LLAVA_1_5_7B: ModelConfig(
            pretrained_model_name="llava-hf/llava-1.5-7b-hf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAVA_1_5_7B

    sample_image = "https://www.ilankelman.org/stopsigns/australia.jpg"
    sample_text = "What’s shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize LLaVA model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="llava",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, dtype_override=None):
        """Load and return the LLaVA model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = LlavaForConditionalGeneration.from_pretrained(str(model_name))
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for LLaVA."""
        if self.processor is None:
            self._load_processor()

        # Build prompt
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

        # Load image

        ## Add the get file utililty here
        input_image = get_file("https://www.ilankelman.org/stopsigns/australia.jpg")
        image = Image.open(str(input_image))

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
