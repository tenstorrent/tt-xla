# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Video-LLaVA model loader implementation for multimodal conditional generation.
"""

from typing import Optional

import numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

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
    """Available Video-LLaVA model variants."""

    VIDEO_LLAVA_7B = "Video-LLaVA-7B"


class ModelLoader(ForgeModel):
    """Video-LLaVA model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.VIDEO_LLAVA_7B: ModelConfig(
            pretrained_model_name="LanguageBind/Video-LLaVA-7B-hf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIDEO_LLAVA_7B

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Video-LLaVA model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Video-LLaVA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        model_name = self._variant_config.pretrained_model_name
        self.processor = VideoLlavaProcessor.from_pretrained(model_name)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Video-LLaVA model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            str(model_name), **kwargs
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for Video-LLaVA."""
        if self.processor is None:
            self._load_processor()

        text_prompt = "USER: <video>\nWhat is shown in this video? ASSISTANT:"

        # Create a small synthetic video (8 frames of 32x32 RGB)
        video = np.random.randint(0, 255, (8, 32, 32, 3), dtype=np.uint8)

        inputs = self.processor(text=text_prompt, videos=video, return_tensors="pt")

        if dtype_override:
            inputs = {
                k: cast_input_to_type(v, dtype_override) for k, v in inputs.items()
            }

        return dict(inputs)
