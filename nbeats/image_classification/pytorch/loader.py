# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
N-BEATS model loader implementation for time series forecasting.
"""

from typing import Optional
from dataclasses import dataclass

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from .src.dataset import (
    get_electricity_dataset_input,
)
from .src.model import (
    NBeatsWithGenericBasis,
    NBeatsWithSeasonalityBasis,
    NBeatsWithTrendBasis,
)


@dataclass
class NBeatsConfig(ModelConfig):
    input_size: int
    output_size: int
    stacks: int
    layers: int
    layer_size: int
    degree_of_polynomial: Optional[int] = None
    num_of_harmonics: Optional[int] = None


class ModelVariant(StrEnum):
    GENERIC_BASIS = "generic_basis"
    SEASONALITY_BASIS = "seasonality_basis"
    TREND_BASIS = "trend_basis"


class ModelLoader(ForgeModel):
    """N-BEATS model loader implementation supporting multiple basis variants."""

    _VARIANTS = {
        ModelVariant.GENERIC_BASIS: NBeatsConfig(
            pretrained_model_name="generic_basis",
            input_size=72,
            output_size=24,
            stacks=30,
            layers=4,
            layer_size=512,
        ),
        ModelVariant.SEASONALITY_BASIS: NBeatsConfig(
            pretrained_model_name="seasonality_basis",
            input_size=72,
            output_size=24,
            stacks=30,
            layers=4,
            layer_size=2048,
            num_of_harmonics=1,
        ),
        ModelVariant.TREND_BASIS: NBeatsConfig(
            pretrained_model_name="trend_basis",
            input_size=72,
            output_size=24,
            stacks=30,
            layers=4,
            layer_size=256,
            degree_of_polynomial=3,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GENERIC_BASIS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="nbeats",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the N-BEATS model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The N-BEATS model instance for time series forecasting.
        """
        cfg = self._variant_config

        if self._variant == ModelVariant.GENERIC_BASIS:
            model = NBeatsWithGenericBasis(
                input_size=cfg.input_size,
                output_size=cfg.output_size,
                stacks=cfg.stacks,
                layers=cfg.layers,
                layer_size=cfg.layer_size,
            )
        elif self._variant == ModelVariant.SEASONALITY_BASIS:
            model = NBeatsWithSeasonalityBasis(
                input_size=cfg.input_size,
                output_size=cfg.output_size,
                num_of_harmonics=cfg.num_of_harmonics or 1,
                stacks=cfg.stacks,
                layers=cfg.layers,
                layer_size=cfg.layer_size,
            )
        elif self._variant == ModelVariant.TREND_BASIS:
            model = NBeatsWithTrendBasis(
                input_size=cfg.input_size,
                output_size=cfg.output_size,
                degree_of_polynomial=cfg.degree_of_polynomial or 3,
                stacks=cfg.stacks,
                layers=cfg.layers,
                layer_size=cfg.layer_size,
            )
        else:
            raise ValueError(f"Unsupported variant: {self._variant}")

        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the N-BEATS model with default settings.

        Returns:
            list: Input tensors and attention masks that can be fed to the model.
        """
        x, x_mask = get_electricity_dataset_input()

        if dtype_override is not None:
            x = x.to(dtype_override)
            x_mask = x_mask.to(dtype_override)

        return [x, x_mask]
