# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaVA-Video model loader implementation for multimodal conditional generation.
"""

from typing import Optional

import numpy as np
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available LLaVA-Video model variants."""

    LLAVA_VIDEO_7B_QWEN2 = "7B_Qwen2"


class ModelLoader(ForgeModel):
    """LLaVA-Video model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.LLAVA_VIDEO_7B_QWEN2: ModelConfig(
            pretrained_model_name="lmms-lab/LLaVA-Video-7B-Qwen2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAVA_VIDEO_7B_QWEN2

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize LLaVA-Video model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LLaVA-Video",
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
        """Load and return the LLaVA-Video model instance."""
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
        """Load and return input tensors for LLaVA-Video."""
        if self.processor is None:
            self._load_processor()

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": "Describe this video in detail."},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        # Create a small synthetic video (8 frames of 32x32 RGB)
        video = np.random.randint(0, 255, (8, 32, 32, 3), dtype=np.uint8)

        inputs = self.processor(text=text_prompt, videos=[video], return_tensors="pt")

        if dtype_override:
            inputs = {
                k: cast_input_to_type(v, dtype_override) for k, v in inputs.items()
            }

        return dict(inputs)
