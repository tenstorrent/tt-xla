# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OlmoEarth feature extraction model loader implementation.

OlmoEarth-v1-Base is a ViT-Base (89M parameters) foundation model for remote
sensing tasks over Sentinel-1, Sentinel-2, and Landsat images or image time series.
Uses the olmoearth-pretrain library for model loading.
"""

import torch
from typing import Optional

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

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
    """Available OlmoEarth model variants for feature extraction."""

    V1_BASE = "v1_base"


class ModelLoader(ForgeModel):
    """OlmoEarth model loader for satellite image feature extraction."""

    _VARIANTS = {
        ModelVariant.V1_BASE: ModelConfig(
            pretrained_model_name="allenai/OlmoEarth-v1-Base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1_BASE

    # Map variant to olmoearth_pretrain ModelID
    _MODEL_IDS = {
        ModelVariant.V1_BASE: ModelID.OLMOEARTH_V1_BASE,
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="olmo_earth",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the OlmoEarth model encoder.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The OlmoEarth encoder model.
        """
        model_id = self._MODEL_IDS[self._variant]
        model = load_model_from_id(model_id)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        self.model = model.encoder
        return self.model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the OlmoEarth model.

        Creates synthetic Sentinel-2 L2A imagery inputs in BHWTC format.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Batch size for the inputs.

        Returns:
            MaskedOlmoEarthSample: Input sample for the model encoder.
        """
        # Sentinel-2 L2A has 12 bands, using 64x64 spatial and 1 timestep
        image = torch.randn(batch_size, 64, 64, 1, 12)
        mask = torch.ones(batch_size, 64, 64, 1, 3) * MaskValue.ONLINE_ENCODER.value
        timestamps = torch.tensor([[[15, 6, 2024]]]).expand(batch_size, -1, -1)

        if dtype_override is not None:
            image = image.to(dtype_override)

        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=image,
            sentinel2_l2a_mask=mask,
            timestamps=timestamps,
        )

        return sample
