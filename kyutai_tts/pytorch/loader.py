# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kyutai TTS 1.6B streaming text-to-speech model loader implementation.
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

    TTS_1_6B_EN_FR = "TTS 1.6B en_fr"


class ModelLoader(ForgeModel):
    """Kyutai TTS 1.6B streaming text-to-speech model loader implementation."""

    _VARIANTS = {
        ModelVariant.TTS_1_6B_EN_FR: ModelConfig(
            pretrained_model_name="kyutai/tts-1.6b-en_fr",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TTS_1_6B_EN_FR

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="KyutaiTTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kyutai TTS LM backbone."""
        from moshi.models.loaders import CheckpointInfo
        from moshi.models.tts import TTSModel

        pretrained_model_name = self._variant_config.pretrained_model_name

        checkpoint_info = CheckpointInfo.from_hf_repo(pretrained_model_name)
        dtype = dtype_override if dtype_override is not None else torch.float32
        tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info, n_q=32, temp=0.6, device=torch.device("cpu"), dtype=dtype
        )
        model = tts_model.lm_gen.model
        model.eval()

        self._num_codebooks = model.num_codebooks
        self._card = model.card

        return model

    def load_inputs(self, dtype_override=None):
        """Load synthetic discrete code inputs for the Kyutai TTS model.

        The model expects discrete codes of shape [B, K, T] where
        K is the number of codebooks and T is time steps.
        """
        codes = torch.randint(0, self._card, (1, self._num_codebooks, 10))
        return codes
