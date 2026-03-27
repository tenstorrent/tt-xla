# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GigaAM-v3 model loader implementation for automatic speech recognition.

GigaAM-v3 is a Conformer-based ASR foundation model pretrained on 700k hours
of Russian speech. It supports CTC and RNN-T decoding variants.
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
    """Available GigaAM-v3 speech recognition model variants."""

    CTC = "CTC"
    RNNT = "RNNT"


class ModelLoader(ForgeModel):
    """GigaAM-v3 model loader implementation for automatic speech recognition."""

    _VARIANTS = {
        ModelVariant.CTC: ModelConfig(
            pretrained_model_name="ai-sage/GigaAM-v3",
        ),
        ModelVariant.RNNT: ModelConfig(
            pretrained_model_name="ai-sage/GigaAM-v3",
        ),
    }

    # Map variant enum to HuggingFace revision branch
    _REVISION_MAP = {
        ModelVariant.CTC: "ctc",
        ModelVariant.RNNT: "rnnt",
    }

    DEFAULT_VARIANT = ModelVariant.CTC

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GigaAM-v3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        revision = self._REVISION_MAP[self._variant]

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            revision=revision,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        # GigaAM-v3 expects raw audio at 16kHz sample rate
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)

        if dtype_override is not None:
            audio_tensor = audio_tensor.to(dtype_override)

        return [audio_tensor]
