# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ultravox model loader implementation for speech language modeling.
"""

import numpy as np
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
    """Available Ultravox model variants."""

    V0_5_LLAMA_3_2_1B = "v0_5_Llama_3_2_1B"


class ModelLoader(ForgeModel):
    """Ultravox model loader implementation for speech language modeling tasks."""

    _VARIANTS = {
        ModelVariant.V0_5_LLAMA_3_2_1B: ModelConfig(
            pretrained_model_name="fixie-ai/ultravox-v0_5-llama-3_2-1b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V0_5_LLAMA_3_2_1B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Ultravox",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant."""
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Ultravox model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Ultravox model instance.
        """
        import transformers

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = transformers.AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Ultravox model.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        # Generate a synthetic 3-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 3
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        turns = [
            {
                "role": "user",
                "content": "Describe what you hear in this audio.",
            },
        ]

        text = self.processor.tokenizer.apply_chat_template(
            turns, add_generation_prompt=True, tokenize=False
        )

        inputs = self.processor(
            text=text,
            audio=audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        return inputs
