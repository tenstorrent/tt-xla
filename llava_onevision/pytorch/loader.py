# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaVA-OneVision model loader implementation for multimodal conditional generation.
"""

import torch
from PIL import Image
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
from typing import Optional

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
from ...tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available LLaVA-OneVision model variants."""

    LLAVA_ONEVISION_QWEN2_0_5B_OV = "Qwen2_0.5B_OV"
    LLAVA_ONEVISION_QWEN2_7B_SI = "Qwen2_7B_SI"


class ModelLoader(ForgeModel):
    """LLaVA-OneVision model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.LLAVA_ONEVISION_QWEN2_0_5B_OV: ModelConfig(
            pretrained_model_name="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        ),
        ModelVariant.LLAVA_ONEVISION_QWEN2_7B_SI: ModelConfig(
            pretrained_model_name="llava-hf/llava-onevision-qwen2-7b-si-hf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAVA_ONEVISION_QWEN2_0_5B_OV

    sample_text = "What are these?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize LLaVA-OneVision model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LLaVA-OneVision",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LLaVA-OneVision model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            str(model_name), **kwargs
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for LLaVA-OneVision."""
        if self.processor is None:
            self._load_processor()

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
            conversation, add_generation_prompt=True
        )

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)

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
