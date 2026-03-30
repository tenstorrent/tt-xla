# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pyannote Segmentation 3.0 model loader implementation.

Loads the onnx-community/pyannote-segmentation-3.0 model using transformers'
AutoModelForAudioFrameClassification for speaker segmentation tasks.
"""

from typing import Optional

import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Pyannote Segmentation 3.0 model variants."""

    SEGMENTATION_3_0 = "Segmentation_3_0"


class ModelLoader(ForgeModel):
    """Pyannote Segmentation 3.0 model loader implementation."""

    _VARIANTS = {
        ModelVariant.SEGMENTATION_3_0: ModelConfig(
            pretrained_model_name="onnx-community/pyannote-segmentation-3.0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEGMENTATION_3_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.feature_extractor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="PyannoteSegmentation3_0",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Pyannote Segmentation 3.0 model from onnx-community."""
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = AutoModelForAudioFrameClassification.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.model

    def load_inputs(self, dtype_override=None):
        """Generate sample audio inputs for the segmentation model.

        Returns a 10-second mono audio waveform at 16kHz, processed through
        the model's feature extractor.
        """
        if self.feature_extractor is None:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        dtype = dtype_override or torch.float32
        # 10 seconds of mono audio at 16kHz
        waveform = torch.randn(160000, dtype=dtype)

        inputs = self.feature_extractor(
            waveform, sampling_rate=16000, return_tensors="pt"
        )

        return [inputs["input_values"]]
