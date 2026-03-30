# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TerraMind model loader implementation for geospatial feature extraction.

TerraMind is a multimodal any-to-any generative foundation model for Earth
Observation developed by IBM and ESA. It uses a dual-scale transformer-based
encoder-decoder architecture and is loaded via the TerraTorch library.
"""

import torch
from typing import Optional

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
    """Available TerraMind model variants."""

    LARGE = "large"


class ModelLoader(ForgeModel):
    """TerraMind model loader for geospatial feature extraction tasks."""

    _VARIANTS = {
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="ibm-esa-geospatial/TerraMind-1.0-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="TerraMind",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the TerraMind model as a backbone feature extractor.

        Requires the ``terratorch`` package to be installed.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The TerraMind backbone model instance.
        """
        from terratorch import BACKBONE_REGISTRY

        model = BACKBONE_REGISTRY.build(
            "terramind_v1_large",
            pretrained=True,
            modalities=["S2L2A"],
        )

        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the TerraMind model.

        Generates a synthetic Sentinel-2 L2A input tensor with 12 spectral
        bands at 224x224 resolution.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Number of samples in the batch.

        Returns:
            dict: Dictionary mapping modality name to input tensor.
        """
        # Sentinel-2 L2A has 12 spectral bands, expected input size is 224x224
        inputs = {"S2L2A": torch.randn(batch_size, 12, 224, 224)}

        if dtype_override is not None:
            inputs = {k: v.to(dtype_override) for k, v in inputs.items()}

        return inputs
