# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Braindecode BENDR model loader implementation for EEG classification.

BENDR (BErt-inspired Neural Data Representations) is a pretrained
transformer-based model for EEG classification tasks using self-supervised
learning on masked sequence reconstruction.
"""

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
    """Available Braindecode BENDR model variants."""

    BENDR_PRETRAINED = "bendr_pretrained"


class ModelLoader(ForgeModel):
    """Braindecode BENDR model loader implementation for EEG classification (PyTorch)."""

    _VARIANTS = {
        ModelVariant.BENDR_PRETRAINED: ModelConfig(
            pretrained_model_name="braindecode/braindecode-bendr",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BENDR_PRETRAINED

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="BENDR",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from braindecode.models import BENDR
        from huggingface_hub import hf_hub_download
        import torch

        model = BENDR(n_chans=20, n_outputs=2)

        checkpoint_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="pytorch_model.bin",
        )
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        # Handle both raw state_dict and wrapped {"model_state_dict": ...} formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)

        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import torch

        # BENDR expects raw EEG data: (batch, channels, time_samples)
        # 20 EEG channels, 600 time samples (~1 second at typical sampling rates)
        n_chans = 20
        n_samples = 600
        eeg_data = torch.randn(1, n_chans, n_samples)

        if dtype_override is not None:
            eeg_data = eeg_data.to(dtype_override)

        return [eeg_data]
