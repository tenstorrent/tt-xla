# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SignalJEPA Contextual model loader implementation for EEG signal classification.

SignalJEPA is a Joint Embedding Predictive Architecture for cross-dataset
transfer learning on EEG (electroencephalography) data. It uses spatial
attention mechanisms to handle varying electrode configurations across datasets.
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
    """Available SignalJEPA model variants."""

    CONTEXTUAL_PRETRAINED = "Contextual_Pretrained"


class ModelLoader(ForgeModel):
    """SignalJEPA Contextual model loader for EEG signal classification."""

    _VARIANTS = {
        ModelVariant.CONTEXTUAL_PRETRAINED: ModelConfig(
            pretrained_model_name="braindecode/SignalJEPA-Contextual-pretrained",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTEXTUAL_PRETRAINED

    # EEG configuration: 19 channels (10-20 system), 128 Hz, 2-second windows
    _N_CHANNELS = 19
    _N_TIMES = 256

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SignalJEPA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the SignalJEPA Contextual model.

        Returns:
            torch.nn.Module: The SignalJEPA_Contextual model instance.
        """
        from braindecode.models import SignalJEPA_Contextual

        cfg = self._variant_config

        model = SignalJEPA_Contextual.from_pretrained(cfg.pretrained_model_name)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample EEG inputs for the model.

        Returns:
            torch.Tensor: Input tensor of shape (batch, n_channels, n_times).
        """
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        inputs = torch.randn(1, self._N_CHANNELS, self._N_TIMES, dtype=dtype)

        return inputs
