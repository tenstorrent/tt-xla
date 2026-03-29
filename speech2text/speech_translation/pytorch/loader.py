# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Speech2Text model loader implementation for speech translation.
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
    """Available Speech2Text model variants for speech translation."""

    SMALL_MUSTC_EN_FR_ST = "Small_Mustc_En_Fr_St"


class ModelLoader(ForgeModel):
    """Speech2Text model loader implementation for speech translation."""

    _VARIANTS = {
        ModelVariant.SMALL_MUSTC_EN_FR_ST: ModelConfig(
            pretrained_model_name="facebook/s2t-small-mustc-en-fr-st",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMALL_MUSTC_EN_FR_ST

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self._processor = None
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        return ModelInfo(
            model="Speech2Text",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant."""
        from transformers import Speech2TextProcessor

        self._processor = Speech2TextProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Speech2Text model for this instance's variant."""
        from transformers import Speech2TextForConditionalGeneration

        if self._processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Speech2TextForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Speech2Text model."""
        import numpy as np

        if self._processor is None:
            self._load_processor()

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        decoder_start_token_id = self._model.config.decoder_start_token_id
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
