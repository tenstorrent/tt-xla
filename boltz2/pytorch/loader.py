#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Boltz2 model loader implementation
"""

from typing import Optional
import torch

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

from third_party.tt_forge_models.boltz2.pytorch.src.model_utils import (
    load_boltz2_inputs,
    load_boltz2_model,
)


class ModelVariant(StrEnum):
    """Available Boltz model variants."""

    BOLTZ2 = "boltz2"


class ModelLoader(ForgeModel):
    """Boltz model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BOLTZ2: ModelConfig(
            pretrained_model_name="boltz2",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BOLTZ2

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="boltz",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.model: Optional[torch.nn.Module] = None

    def load_model(self, dtype_override=None):
        return load_boltz2_model()

    def load_inputs(self, dtype_override: Optional[torch.dtype] = torch.float32):
        return load_boltz2_inputs(dtype_override=dtype_override)
