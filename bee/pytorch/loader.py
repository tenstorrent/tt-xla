# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bee model loader implementation for multimodal conditional generation.
"""

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
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
    """Available Bee model variants."""

    BEE_8B_RL = "Bee_8B_RL"


class ModelLoader(ForgeModel):
    """Bee model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.BEE_8B_RL: ModelConfig(
            pretrained_model_name="Open-Bee/Bee-8B-RL",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BEE_8B_RL

    sample_text = "What are these?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Bee model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Bee",
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
        """Load and return the Bee model instance."""
        model_name = self._variant_config.pretrained_model_name
        kwargs.setdefault("trust_remote_code", True)
        model = AutoModel.from_pretrained(str(model_name), **kwargs)
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for Bee."""
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

        if dtype_override:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return dict(inputs)
