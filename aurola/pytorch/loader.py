# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AuroLA-3B model loader implementation for audio-text retrieval.
"""
import numpy as np
import torch
from transformers import (
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
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


class ModelVariant(StrEnum):
    """Available AuroLA model variants."""

    AUROLA_3B = "3B"


class ModelLoader(ForgeModel):
    """AuroLA-3B model loader implementation for audio-text retrieval tasks."""

    _VARIANTS = {
        ModelVariant.AUROLA_3B: ModelConfig(
            pretrained_model_name="Jazzcharles/AuroLA-3B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AUROLA_3B

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="AuroLA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant."""
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the AuroLA-3B model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.config.use_cache = False
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the AuroLA-3B model."""
        if self.processor is None:
            self._load_processor()

        # Generate a synthetic audio waveform (sine wave at 440Hz, 1 second)
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio_waveform = np.sin(2 * np.pi * 440 * t)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_waveform},
                    {"type": "text", "text": "Describe this audio."},
                ],
            }
        ]

        from qwen_omni_utils import process_mm_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

        inputs = self.processor(
            text=[text],
            audios=audios,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].is_floating_point():
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
