# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaVA-Llama-3 model loader implementation for multimodal conditional generation.
"""

from typing import Optional

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
from ...tools.utils import get_file, cast_input_to_type


class ModelVariant(StrEnum):
    """Available LLaVA-Llama-3 model variants."""

    LLAVA_LLAMA3_8B_V1_1 = "Llama3_8B_v1.1"


class ModelLoader(ForgeModel):
    """LLaVA-Llama-3 model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.LLAVA_LLAMA3_8B_V1_1: ModelConfig(
            pretrained_model_name="xtuner/llava-llama-3-8b-v1_1-transformers",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAVA_LLAMA3_8B_V1_1

    sample_text = "What are these?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize LLaVA-Llama-3 model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LLaVA-Llama-3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LLaVA-Llama-3 model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = LlavaForConditionalGeneration.from_pretrained(str(model_name), **kwargs)
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for LLaVA-Llama-3."""
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

        # Load sample image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)

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
