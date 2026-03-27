# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VideoLLaMA3-7B model loader implementation for multimodal video understanding.
"""

from typing import Optional

import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor

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
    """Available VideoLLaMA3 model variants."""

    BASE_7B = "base_7b"


class ModelLoader(ForgeModel):
    """VideoLLaMA3-7B model loader for multimodal video understanding."""

    _VARIANTS = {
        ModelVariant.BASE_7B: ModelConfig(
            pretrained_model_name="DAMO-NLP-SG/VideoLLaMA3-7B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_7B

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize VideoLLaMA3 model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="VideoLLaMA3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        model_name = self._variant_config.pretrained_model_name
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the VideoLLaMA3 model instance."""
        model_name = self._variant_config.pretrained_model_name
        kwargs.setdefault("trust_remote_code", True)
        model = AutoModelForCausalLM.from_pretrained(str(model_name), **kwargs)
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for VideoLLaMA3."""
        if self.processor is None:
            self._load_processor()

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"num_frames": 8}},
                    {
                        "type": "text",
                        "text": "Describe what is happening in this video.",
                    },
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        # Create a small synthetic video (8 frames of 32x32 RGB)
        video = np.random.randint(0, 255, (8, 32, 32, 3), dtype=np.uint8)

        inputs = self.processor(text=text, videos=[video], return_tensors="pt")

        if dtype_override:
            inputs = {
                k: cast_input_to_type(v, dtype_override) for k, v in inputs.items()
            }

        return dict(inputs)
