# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TimesFM 2.0 model loader implementation for time series forecasting.
"""

from typing import Optional

import torch

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


class ModelVariant(StrEnum):
    """Available TimesFM model variants."""

    TIMESFM_2_500M = "TimesFM_2_500M"


class ModelLoader(ForgeModel):
    """TimesFM 2.0 model loader implementation for time series forecasting."""

    _VARIANTS = {
        ModelVariant.TIMESFM_2_500M: ModelConfig(
            pretrained_model_name="google/timesfm-2.0-500m-pytorch",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TIMESFM_2_500M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="TimesFM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the TimesFM 2.0 model."""
        import timesfm

        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=32,
                horizon_len=128,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=50,
                model_dims=1280,
                use_positional_embedding=False,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self._variant_config.pretrained_model_name,
            ),
        )

        return tfm

    def load_inputs(self, dtype_override=None):
        """Load sample time series input for the TimesFM model.

        Returns a batch of synthetic time series context data.
        """
        import numpy as np

        # TimesFM expects a list of 1D arrays as context
        torch.manual_seed(42)
        context = torch.randn(512)

        if dtype_override is not None:
            context = context.to(dtype_override)

        return {"context": context}
