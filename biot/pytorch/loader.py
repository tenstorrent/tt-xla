# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BIOT (Brain-Inspired Optimal Transformer) model loader for biosignal classification.
"""

import torch
from typing import Optional
from dataclasses import dataclass

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


@dataclass
class BiotConfig(ModelConfig):
    n_chans: int = 18
    n_times: int = 2000
    sfreq: int = 200


class ModelVariant(StrEnum):
    """Available BIOT model variants."""

    SHHS_PREST_18CHS = "shhs_prest_18chs"


class ModelLoader(ForgeModel):
    """BIOT model loader for biosignal feature extraction."""

    _VARIANTS = {
        ModelVariant.SHHS_PREST_18CHS: BiotConfig(
            pretrained_model_name="braindecode/biot-pretrained-shhs-prest-18chs",
            n_chans=18,
            n_times=2000,
            sfreq=200,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SHHS_PREST_18CHS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="BIOT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from braindecode.models import BIOT

        cfg = self._variant_config

        model = BIOT.from_pretrained(
            cfg.pretrained_model_name,
            n_chans=cfg.n_chans,
            n_times=cfg.n_times,
            sfreq=cfg.sfreq,
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        inputs = torch.randn(1, cfg.n_chans, cfg.n_times, dtype=dtype)

        return inputs
