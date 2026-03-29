# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MatAnyone model loader implementation for video matting tasks.
"""

import torch
import numpy as np
from typing import Optional
from matanyone.model.matanyone import MatAnyone

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
    """Available MatAnyone model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """MatAnyone video matting model loader implementation."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="PeiqingYang/MatAnyone",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MatAnyone",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MatAnyone model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = MatAnyone.from_pretrained(pretrained_model_name)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load synthetic inputs for the MatAnyone model.

        MatAnyone expects an image tensor and a corresponding binary mask
        for video matting. We generate synthetic inputs for testing.
        """
        # MatAnyone uses ImageNet normalization internally
        # Input image: (B, 3, H, W) float tensor
        # Input mask: (B, 1, H, W) float tensor with values in {0, 1}
        height, width = 480, 640
        image = torch.from_numpy(
            np.random.randn(batch_size, 3, height, width).astype(np.float32)
        )
        mask = torch.from_numpy(
            np.random.choice([0.0, 1.0], size=(batch_size, 1, height, width)).astype(
                np.float32
            )
        )

        if dtype_override is not None:
            image = image.to(dtype_override)
            mask = mask.to(dtype_override)

        return {"image": image, "mask": mask}
