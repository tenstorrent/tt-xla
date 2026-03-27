# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Facebook SAM Audio Judge model loader for audio quality evaluation.
"""

from typing import Optional

import torch

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available SAM Audio Judge model variants."""

    DEFAULT = "Default"


class SAMAudioJudgeWrapper(torch.nn.Module):
    """Wrapper module for the SAM Audio Judge model.

    Wraps the SAMAudioJudgeModel to accept preprocessed tensor inputs
    directly, making it compatible with the ForgeModel interface.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        output = self.model(**kwargs)
        return output.overall


class ModelLoader(ForgeModel):
    """Facebook SAM Audio Judge model loader for audio quality evaluation."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="facebook/sam-audio-judge",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SAMAudioJudge",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_AUDIO_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from sam_audio import SAMAudioJudgeProcessor

        self._processor = SAMAudioJudgeProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from sam_audio import SAMAudioJudgeModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = SAMAudioJudgeModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return SAMAudioJudgeWrapper(model)

    def load_inputs(self, dtype_override=None):
        if self._processor is None:
            self._load_processor()

        # Generate synthetic 1-second audio waveforms at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        num_samples = sampling_rate * duration_seconds

        input_audio = torch.randn(1, num_samples)
        separated_audio = torch.randn(1, num_samples)
        description = "A person speaking"

        inputs = self._processor(
            text=[description],
            input_audio=[input_audio],
            separated_audio=[separated_audio],
        )

        if dtype_override is not None:
            inputs = {
                k: v.to(dtype_override) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

        return inputs
