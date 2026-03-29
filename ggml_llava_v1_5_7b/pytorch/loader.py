# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GGML LLaVA v1.5 7B GGUF model loader implementation for multimodal conditional generation.
"""

from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
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
    """Available GGML LLaVA v1.5 7B GGUF model variants."""

    GGML_LLAVA_V1_5_7B_Q4_K = "v1.5_7B_Q4_K"


class ModelLoader(ForgeModel):
    """GGML LLaVA v1.5 7B GGUF model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.GGML_LLAVA_V1_5_7B_Q4_K: ModelConfig(
            pretrained_model_name="mys/ggml_llava-v1.5-7b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GGML_LLAVA_V1_5_7B_Q4_K

    GGUF_FILE = "ggml-model-q4_k.gguf"

    PROCESSOR_MODEL = "llava-hf/llava-1.5-7b-hf"

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GGML LLaVA v1.5 7B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.PROCESSOR_MODEL)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGML LLaVA v1.5 7B GGUF model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        model = LlavaForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for GGML LLaVA v1.5 7B GGUF."""
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

        if dtype_override:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)
            pixel_values = cast_input_to_type(pixel_values, dtype_override)

        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
