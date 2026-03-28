# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
OWSM CTC v4 model loader implementation for speech recognition (ASR) using PyTorch.

ESPnet-based multilingual speech recognition model using hierarchical
multi-task self-conditioned CTC (encoder-only, non-autoregressive).
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
    """Available OWSM CTC v4 PyTorch speech recognition model variants."""

    V4_1B = "v4_1B"


class ModelLoader(ForgeModel):
    """OWSM CTC v4 model loader implementation for speech recognition (PyTorch)."""

    _VARIANTS = {
        ModelVariant.V4_1B: ModelConfig(
            pretrained_model_name="espnet/owsm_ctc_v4_1B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V4_1B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._s2t = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="OWSM_CTC_v4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

        self._s2t = Speech2TextGreedySearch.from_pretrained(
            self._variant_config.pretrained_model_name,
            device="cpu",
            lang_sym="<eng>",
            task_sym="<asr>",
        )

        model = self._s2t.s2t_model
        model.eval()

        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        speech = np.random.randn(sampling_rate * duration_seconds).astype(np.float32)

        speech_tensor = torch.from_numpy(speech).unsqueeze(0)
        speech_lengths = torch.tensor([speech_tensor.shape[1]], dtype=torch.long)

        if dtype_override is not None:
            speech_tensor = speech_tensor.to(dtype_override)

        return {"speech": speech_tensor, "speech_lengths": speech_lengths}
