# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Moirai model loader implementation for time series forecasting.
"""

from dataclasses import dataclass
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


@dataclass
class MoiraiConfig(ModelConfig):
    context_length: int = 512
    prediction_length: int = 64
    patch_size: int = 32
    target_dim: int = 1
    num_samples: int = 100


class ModelVariant(StrEnum):
    """Available Moirai model variants."""

    BASE_1_0 = "base_1_0"
    LARGE = "large"
    LARGE_1_0 = "large_1_0"


class ModelLoader(ForgeModel):
    """Moirai model loader for time series forecasting.

    Uses the uni2ts MoiraiForecast wrapper around MoiraiModule
    for univariate time series prediction.
    """

    _VARIANTS = {
        ModelVariant.BASE_1_0: MoiraiConfig(
            pretrained_model_name="Salesforce/moirai-1.0-R-base",
            context_length=512,
            prediction_length=64,
            patch_size=32,
            target_dim=1,
            num_samples=100,
        ),
        ModelVariant.LARGE: MoiraiConfig(
            pretrained_model_name="Salesforce/moirai-1.1-R-large",
            context_length=512,
            prediction_length=64,
            patch_size=32,
            target_dim=1,
            num_samples=100,
        ),
        ModelVariant.LARGE_1_0: MoiraiConfig(
            pretrained_model_name="Salesforce/moirai-1.0-R-large",
            context_length=512,
            prediction_length=64,
            patch_size=32,
            target_dim=1,
            num_samples=100,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Moirai",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Moirai forecasting model.

        Returns:
            torch.nn.Module: MoiraiForecast instance.
        """
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

        cfg = self._variant_config

        module = MoiraiModule.from_pretrained(cfg.pretrained_model_name)

        model = MoiraiForecast(
            module=module,
            prediction_length=cfg.prediction_length,
            context_length=cfg.context_length,
            patch_size=cfg.patch_size,
            target_dim=cfg.target_dim,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            num_samples=cfg.num_samples,
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the Moirai model.

        Returns:
            dict: Input tensors matching MoiraiForecast.forward signature.
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)

        # past_target: (batch, context_length, target_dim)
        past_target = torch.randn(1, cfg.context_length, cfg.target_dim, dtype=dtype)
        # past_observed_target: (batch, context_length, target_dim)
        past_observed_target = torch.ones(
            1, cfg.context_length, cfg.target_dim, dtype=torch.bool
        )
        # past_is_pad: (batch, context_length)
        past_is_pad = torch.zeros(1, cfg.context_length, dtype=torch.bool)

        return {
            "past_target": past_target,
            "past_observed_target": past_observed_target,
            "past_is_pad": past_is_pad,
        }
