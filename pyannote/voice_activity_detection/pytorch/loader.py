# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pyannote voice activity detection model loader implementation.

Loads the voice-activity-detection pipeline and extracts its segmentation
model for testing, as this is the primary neural network component.
"""

import os

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
    """Available Pyannote voice activity detection model variants."""

    VAD = "VAD"


class ModelLoader(ForgeModel):
    """Pyannote voice activity detection model loader implementation.

    Loads the voice-activity-detection pipeline and extracts its
    segmentation model for testing.
    """

    _VARIANTS = {
        ModelVariant.VAD: ModelConfig(
            pretrained_model_name="pyannote/voice-activity-detection",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VAD

    def __init__(self, variant=None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Pyannote",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Pyannote voice activity detection pipeline's segmentation model.

        Requires a HuggingFace token with access to the gated model.
        Set the HF_TOKEN environment variable or pass token as a kwarg.
        """
        from pyannote.audio import Pipeline

        pipeline_kwargs = {}
        token = kwargs.pop("token", None) or os.environ.get("HF_TOKEN")
        if token:
            pipeline_kwargs["token"] = token

        pipeline = Pipeline.from_pretrained(
            self._variant_config.pretrained_model_name, **pipeline_kwargs
        )

        # Extract the segmentation model from the pipeline
        self._model = pipeline._segmentation.model
        self._model.eval()
        if dtype_override is not None:
            self._model.to(dtype_override)
        return self._model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the VAD segmentation model.

        Generates a 10-second mono audio waveform at 16kHz as expected
        by the model: shape (batch_size, num_channels, num_samples) = (1, 1, 160000).
        """
        dtype = dtype_override or torch.float32
        # 10 seconds of mono audio at 16kHz
        waveform = torch.randn(1, 1, 160000, dtype=dtype)
        return [waveform]
