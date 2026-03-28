# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kyutai TTS streaming text-to-speech model loader implementation.
"""

import torch
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
    """Available Kyutai TTS model variants."""

    TTS_0_75B_EN = "TTS 0.75B EN"


class ModelLoader(ForgeModel):
    """Kyutai TTS streaming text-to-speech model loader implementation."""

    _VARIANTS = {
        ModelVariant.TTS_0_75B_EN: ModelConfig(
            pretrained_model_name="kyutai/tts-0.75b-en-public",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TTS_0_75B_EN
    DEFAULT_TEXT = "Hey there! How are you? I had the craziest day today."

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Kyutai TTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kyutai TTS model instance."""
        from moshi.models.loaders import CheckpointInfo
        from moshi.models.tts import TTSModel

        checkpoint_info = CheckpointInfo.from_hf_repo(
            self._variant_config.pretrained_model_name
        )
        device = torch.device("cpu")
        tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info, n_q=16, temp=0.6, cfg_coef=3, device=device
        )

        return tts_model

    def load_inputs(self, dtype_override=None):
        """Load sample text input for the Kyutai TTS model."""
        return (self.DEFAULT_TEXT,)
