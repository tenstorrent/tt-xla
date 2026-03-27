# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TITAN model loader implementation for pathology image feature extraction.
"""

import torch
from transformers import AutoModel
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
    """Available TITAN model variants."""

    TITAN = "TITAN"


class TITANSlideEncoder(torch.nn.Module):
    """Wrapper that calls TITAN's encode_slide_from_patch_features in forward."""

    def __init__(self, titan_model):
        super().__init__()
        self.titan_model = titan_model

    def forward(self, features, coords):
        return self.titan_model.encode_slide_from_patch_features(
            features, coords, patch_size_lv0=512
        )


class ModelLoader(ForgeModel):
    """TITAN model loader for pathology image feature extraction."""

    _VARIANTS = {
        ModelVariant.TITAN: ModelConfig(
            pretrained_model_name="MahmoodLab/TITAN",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TITAN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="TITAN",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the TITAN model instance."""
        model_name = self._variant_config.pretrained_model_name

        kwargs.setdefault("trust_remote_code", True)
        model = AutoModel.from_pretrained(model_name, **kwargs)
        model.eval()

        wrapper = TITANSlideEncoder(model)
        wrapper.eval()

        if dtype_override is not None:
            wrapper = wrapper.to(dtype_override)

        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return synthetic patch feature inputs for TITAN.

        TITAN operates on pre-extracted CONCH v1.5 patch features from
        whole-slide pathology images. We generate synthetic inputs matching
        the expected feature dimensions.
        """
        num_patches = 100
        feature_dim = 512

        features = torch.randn(num_patches, feature_dim)
        coords = torch.randint(0, 10000, (num_patches, 2))

        if dtype_override is not None:
            features = features.to(dtype_override)

        return [features, coords]
