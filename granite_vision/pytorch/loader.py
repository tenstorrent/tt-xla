# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite Vision model loader implementation for multimodal conditional generation.
"""

from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor

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
    """Available Granite Vision model variants."""

    GRANITE_VISION_3_3_2B = "3.3_2B"


class ModelLoader(ForgeModel):
    """Granite Vision model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.GRANITE_VISION_3_3_2B: ModelConfig(
            pretrained_model_name="ibm-granite/granite-vision-3.3-2b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRANITE_VISION_3_3_2B

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Granite Vision model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Granite Vision",
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
        """Load and return the Granite Vision model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = AutoModelForVision2Seq.from_pretrained(
            str(model_name), torch_dtype=torch.float32, **kwargs
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for Granite Vision."""
        if self.processor is None:
            self._load_processor()

        # Load sample image
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        # Build conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        # Process inputs using chat template
        inputs = self.processor.apply_chat_template(
            conversation,
            images=[image],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        if dtype_override:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs
