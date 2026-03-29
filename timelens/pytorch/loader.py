# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TimeLens-8B model loader implementation for video temporal grounding.
"""

from typing import Optional

import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

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


class ModelVariant(StrEnum):
    """Available TimeLens model variants."""

    TIMELENS_8B = "8B"


class ModelLoader(ForgeModel):
    """TimeLens model loader for video temporal grounding."""

    _VARIANTS = {
        ModelVariant.TIMELENS_8B: ModelConfig(
            pretrained_model_name="TencentARC/TimeLens-8B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TIMELENS_8B

    sample_text = "When does the person start speaking in this video?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="TimeLens",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the TimeLens model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs
        )
        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for TimeLens."""
        if self.processor is None:
            self._load_processor()

        # Create a small synthetic video (8 frames of 64x64 RGB)
        video = np.random.randint(0, 255, (8, 64, 64, 3), dtype=np.uint8)
        video_frames = [video[i] for i in range(video.shape[0])]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_frames,
                    },
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            videos=[video],
            padding=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key in inputs:
                if hasattr(inputs[key], "to"):
                    inputs[key] = inputs[key].to(dtype_override)

        return dict(inputs)
