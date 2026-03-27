# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaVA-Llama-3-8B-v1.1 GGUF model loader implementation for multimodal conditional generation.
"""

from typing import Optional

from datasets import load_dataset
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoConfig

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
    """Available LLaVA-Llama-3-8B-v1.1 GGUF model variants."""

    LLAVA_LLAMA_3_8B_V1_1_F16 = "Llama_3_8B_v1.1_F16"


class ModelLoader(ForgeModel):
    """LLaVA-Llama-3-8B-v1.1 GGUF model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.LLAVA_LLAMA_3_8B_V1_1_F16: ModelConfig(
            pretrained_model_name="xtuner/llava-llama-3-8b-v1_1-gguf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAVA_LLAMA_3_8B_V1_1_F16

    _GGUF_FILES = {
        ModelVariant.LLAVA_LLAMA_3_8B_V1_1_F16: "llava-llama-3-8b-v1_1-f16.gguf",
    }

    _PROCESSOR_NAME = "xtuner/llava-llama-3-8b-v1_1-hf"

    sample_text = "What's shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize LLaVA-Llama-3-8B-v1.1 GGUF model loader."""
        super().__init__(variant)
        self.processor = None
        self.config = None

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LLaVA-Llama-3-8B-v1.1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(self._PROCESSOR_NAME)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LLaVA-Llama-3-8B-v1.1 GGUF model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.gguf_file

        model = LlavaForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for LLaVA-Llama-3-8B-v1.1 GGUF."""
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

        # Load dataset
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
