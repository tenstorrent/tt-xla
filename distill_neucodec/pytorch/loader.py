# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Distill-NeuCodec neural audio codec model loader implementation.

Distill-NeuCodec is a distilled FSQ-based 0.8 kbps neural audio codec
that encodes 16kHz speech to discrete codes and decodes back to 24kHz audio.
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
    """Available Distill-NeuCodec model variants."""

    DISTILL_NEUCODEC = "distill_neucodec"


class ModelLoader(ForgeModel):
    """Distill-NeuCodec model loader implementation."""

    _VARIANTS = {
        ModelVariant.DISTILL_NEUCODEC: ModelConfig(
            pretrained_model_name="neuphonic/distill-neucodec",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISTILL_NEUCODEC

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="DistillNeuCodec",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Distill-NeuCodec model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Distill-NeuCodec model instance.
        """
        from neucodec import DistillNeuCodec

        pretrained_model_name = self._variant_config.pretrained_model_name

        model = DistillNeuCodec.from_pretrained(pretrained_model_name, **kwargs)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Distill-NeuCodec model.

        The model expects 16kHz mono audio with shape (batch, channels, time).

        Args:
            dtype_override: Optional torch.dtype to override the input tensor's default dtype.

        Returns:
            torch.Tensor: Audio tensor with shape (1, 1, 16000) representing
                         1 second of 16kHz mono audio.
        """
        # Generate 1 second of random 16kHz mono audio
        audio = torch.randn(1, 1, 16000)

        if dtype_override is not None:
            audio = audio.to(dtype_override)

        return audio
