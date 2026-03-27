# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ContentVec model loader implementation for audio feature extraction.

ContentVec is a HuBERT-based model for self-supervised speech representation
learning, designed to disentangle content information from other variations
in speech such as speaker identity.
"""

import torch
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available ContentVec model variants."""

    BEST = "Best"


class ModelLoader(ForgeModel):
    """ContentVec model loader implementation for audio feature extraction."""

    _VARIANTS = {
        ModelVariant.BEST: ModelConfig(
            pretrained_model_name="lengyue233/content-vec-best",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BEST

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ContentVec",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import HubertModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = HubertModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        dtype = dtype_override or torch.float32
        input_values = torch.tensor(audio_array, dtype=dtype).unsqueeze(0)

        return {"input_values": input_values}
