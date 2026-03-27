# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NeuCodec ONNX Decoder model loader implementation.

NeuCodec is an audio codec optimized for on-device TTS at 0.8 kbps.
This loader handles the ONNX decoder component which reconstructs
24kHz audio from Finite Scalar Quantization (FSQ) codes.
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
    """Available NeuCodec ONNX Decoder model variants."""

    NEUCODEC_ONNX_DECODER = "NeuCodec ONNX Decoder"


class ModelLoader(ForgeModel):
    """NeuCodec ONNX Decoder model loader implementation."""

    _VARIANTS = {
        ModelVariant.NEUCODEC_ONNX_DECODER: ModelConfig(
            pretrained_model_name="neuphonic/neucodec-onnx-decoder",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEUCODEC_ONNX_DECODER

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="NeuCodec ONNX Decoder",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the NeuCodec ONNX Decoder model instance."""
        from neucodec import NeuCodecOnnxDecoder

        pretrained_model_name = self._variant_config.pretrained_model_name
        model = NeuCodecOnnxDecoder.from_pretrained(pretrained_model_name)

        return model

    def load_inputs(self, **kwargs):
        """Load sample FSQ code inputs for the NeuCodec ONNX Decoder.

        The decoder expects FSQ codes produced by the NeuCodec encoder.
        We generate random integer codes as a synthetic input.
        """
        # FSQ codes shape: (batch, time_steps)
        # ~1 second of audio at typical codec frame rate
        batch_size = 1
        num_frames = 150
        fsq_codes = torch.randint(0, 8, (batch_size, num_frames))

        return fsq_codes
