# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Vocos EnCodec 24kHz model loader implementation.

Vocos is a fast neural vocoder that reconstructs audio waveforms from EnCodec tokens
using spectral coefficient prediction and inverse STFT.
"""
import torch
import torch.nn as nn
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


class VocosDecodeWrapper(nn.Module):
    """Wrapper around Vocos that exposes decode as the forward pass.

    The Vocos forward() method expects raw audio and re-encodes it internally.
    For inference from EnCodec tokens, we need codes_to_features + decode.
    This wrapper takes pre-computed features and a bandwidth_id and calls decode.
    """

    def __init__(self, vocos):
        super().__init__()
        self.backbone = vocos.backbone
        self.head = vocos.head

    def forward(self, features, bandwidth_id):
        x = self.backbone(features, bandwidth_id=bandwidth_id)
        audio = self.head(x)
        return audio


class ModelVariant(StrEnum):
    """Available Vocos EnCodec model variants."""

    ENCODEC_24KHZ = "Encodec_24khz"


class ModelLoader(ForgeModel):
    """Vocos EnCodec 24kHz model loader implementation."""

    _VARIANTS = {
        ModelVariant.ENCODEC_24KHZ: ModelConfig(
            pretrained_model_name="charactr/vocos-encodec-24khz",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ENCODEC_24KHZ

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.vocos = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="VocosEncodec",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Vocos EnCodec model wrapped for decode inference.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: Wrapped Vocos model that decodes features to audio.
        """
        from vocos import Vocos

        pretrained_model_name = self._variant_config.pretrained_model_name

        vocos = Vocos.from_pretrained(pretrained_model_name)

        if dtype_override is not None:
            vocos = vocos.to(dtype=dtype_override)

        self.vocos = vocos

        model = VocosDecodeWrapper(vocos)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Vocos EnCodec model.

        Generates random EnCodec tokens, converts them to features via the model's
        codebook embedding, and returns them with a bandwidth_id for decoding.

        Args:
            dtype_override: Optional torch.dtype to override the input tensor's default dtype.

        Returns:
            dict: Input tensors containing features and bandwidth_id.
        """
        # Generate random EnCodec tokens: 8 codebooks, 200 frames
        audio_tokens = torch.randint(low=0, high=1024, size=(8, 200))

        # Convert tokens to features using the model's codebook embedding
        features = self.vocos.codes_to_features(audio_tokens)

        # Select 6 kbps bandwidth
        bandwidth_id = torch.tensor([2])

        if dtype_override is not None:
            features = features.to(dtype_override)

        return {"features": features, "bandwidth_id": bandwidth_id}
