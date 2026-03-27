# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mantis-8M model loader implementation for time series classification.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

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
class MantisConfig(ModelConfig):
    seq_len: int = 512
    num_patches: int = 32


class ModelVariant(StrEnum):
    """Available Mantis model variants."""

    MANTIS_8M = "8M"


class ModelLoader(ForgeModel):
    """Mantis-8M model loader for time series classification.

    Loads the Mantis-8M foundation model for time series feature extraction
    and classification from paris-noah/Mantis-8M on Hugging Face.
    """

    _VARIANTS = {
        ModelVariant.MANTIS_8M: MantisConfig(
            pretrained_model_name="paris-noah/Mantis-8M",
            seq_len=512,
            num_patches=32,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MANTIS_8M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Mantis",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Mantis-8M model.

        Returns:
            torch.nn.Module: Mantis8M model instance.
        """
        from mantis.architecture import Mantis8M

        model = Mantis8M.from_pretrained(self._variant_config.pretrained_model_name)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for Mantis-8M.

        The model expects univariate time series interpolated to length 512
        (must be divisible by num_patches=32).

        Returns:
            torch.Tensor: Input tensor of shape (batch, 1, seq_len).
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)

        # Generate a synthetic univariate time series
        raw_series = torch.randn(1, 1, 100, dtype=dtype)

        # Interpolate to model's expected sequence length
        inputs = F.interpolate(
            raw_series, size=cfg.seq_len, mode="linear", align_corners=False
        )

        return inputs
