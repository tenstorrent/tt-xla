# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Shuka model loader implementation for audio-text-to-text generation.
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
    """Available Shuka model variants."""

    SHUKA_V1 = "shuka_v1"


class ModelLoader(ForgeModel):
    """Shuka model loader implementation for audio-text-to-text generation."""

    _VARIANTS = {
        ModelVariant.SHUKA_V1: ModelConfig(
            pretrained_model_name="sarvamai/shuka_v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SHUKA_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Shuka",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load the Shuka processor/pipeline components."""
        import transformers

        self.processor = transformers.pipeline(
            model=self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            torch_dtype="bfloat16",
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Shuka model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Shuka model instance.
        """
        import transformers

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = transformers.AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        self.model.eval()

        return self.model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Shuka model.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        import torch

        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        model_param = next(self.model.parameters())
        device = model_param.device
        dtype = dtype_override or model_param.dtype

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        turns = [
            {"role": "system", "content": "Respond naturally and informatively."},
            {"role": "user", "content": "<|audio|>"},
        ]

        return {
            "audio": audio_array,
            "turns": turns,
            "sampling_rate": sampling_rate,
        }
