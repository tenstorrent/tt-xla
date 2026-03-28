# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MantisV2 model loader implementation for time series classification.
"""

import torch
from typing import Optional

from mantis import MantisV2

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
    MANTIS_V2 = "MantisV2"


class ModelLoader(ForgeModel):
    """MantisV2 model loader for time series classification.

    Loads the MantisV2 time series classification foundation model
    for zero-shot feature extraction.
    """

    _VARIANTS = {
        ModelVariant.MANTIS_V2: ModelConfig(
            pretrained_model_name="paris-noah/MantisV2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MANTIS_V2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MantisV2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the MantisV2 model.

        Returns:
            torch.nn.Module: The MantisV2 model instance.
        """
        model = MantisV2.from_pretrained(self._variant_config.pretrained_model_name)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the model.

        MantisV2 expects input time series of length 512 (num_patches=32,
        kernel_size=41). Input shape is (batch, channels, length).

        Returns:
            Tuple of input tensors for the model.
        """
        dtype = dtype_override or torch.float32

        # Generate synthetic time series input
        torch.manual_seed(42)
        # Shape: (batch=1, channels=1, length=512)
        x = torch.randn(1, 1, 512, dtype=dtype)

        return (x,)
