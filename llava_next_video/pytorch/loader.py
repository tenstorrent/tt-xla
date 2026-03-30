# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaVA-NeXT-Video model loader implementation for multimodal conditional generation.
"""

from typing import Optional

import numpy as np
from transformers import (
    AutoTokenizer,
    LlavaNextImageProcessor,
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
    LlavaNextVideoVideoProcessor,
)

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
    """Available LLaVA-NeXT-Video model variants."""

    TINY_RANDOM = "tiny_random"
    LLAVA_NEXT_VIDEO_7B = "LLaVA_NeXT_Video_7B"


class ModelLoader(ForgeModel):
    """LLaVA-NeXT-Video model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM: ModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-random-llava-next-video",
        ),
        ModelVariant.LLAVA_NEXT_VIDEO_7B: ModelConfig(
            pretrained_model_name="llava-hf/LLaVA-NeXT-Video-7B-hf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAVA_NEXT_VIDEO_7B

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize LLaVA-NeXT-Video model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LLaVA-NeXT-Video",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        model_name = self._variant_config.pretrained_model_name
        if self._variant == ModelVariant.TINY_RANDOM:
            image_processor = LlavaNextImageProcessor.from_pretrained(model_name)
            video_processor = LlavaNextVideoVideoProcessor.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.processor = LlavaNextVideoProcessor(
                video_processor=video_processor,
                image_processor=image_processor,
                tokenizer=tokenizer,
                patch_size=2,
                vision_feature_select_strategy="default",
            )
        else:
            self.processor = LlavaNextVideoProcessor.from_pretrained(model_name)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LLaVA-NeXT-Video model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            str(model_name), **kwargs
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for LLaVA-NeXT-Video."""
        if self.processor is None:
            self._load_processor()

        if self._variant == ModelVariant.TINY_RANDOM:
            text_prompt = "<video>\nWhat is shown in this video?"
            video = np.random.randint(0, 255, (8, 32, 32, 3), dtype=np.uint8)
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                }
            ]
            text_prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            # Create a synthetic video (8 frames of 224x224 RGB)
            video = np.random.randint(0, 255, (8, 224, 224, 3), dtype=np.uint8)

        inputs = self.processor(text=text_prompt, videos=video, return_tensors="pt")

        if dtype_override:
            inputs = {
                k: cast_input_to_type(v, dtype_override) for k, v in inputs.items()
            }

        return dict(inputs)
