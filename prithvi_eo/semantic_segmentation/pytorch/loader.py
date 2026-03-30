# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Prithvi EO 2.0 for Semantic Segmentation model loader implementation
"""

import torch
from typing import Optional
from huggingface_hub import snapshot_download
from terratorch.cli_tools import LightningInferenceModel

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
    """Available Prithvi EO model variants."""

    V2_300M_TL_SEN1FLOODS11 = "V2_300M_TL_Sen1Floods11"


class ModelLoader(ForgeModel):
    """Prithvi EO model loader implementation for semantic segmentation tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.V2_300M_TL_SEN1FLOODS11: ModelConfig(
            pretrained_model_name="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.V2_300M_TL_SEN1FLOODS11

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._model_path = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="PrithviEO",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _download_model(self):
        """Download model files from HuggingFace.

        Returns:
            str: Path to the downloaded model directory
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        self._model_path = snapshot_download(
            repo_id=pretrained_model_name,
            allow_patterns=["*.pt", "*.yaml", "config.json"],
        )
        return self._model_path

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Prithvi EO model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Prithvi EO model instance for semantic segmentation.
        """
        if self._model_path is None:
            self._download_model()

        config_path = f"{self._model_path}/config.yaml"
        checkpoint_path = f"{self._model_path}/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"

        lightning_model = LightningInferenceModel.from_config(
            config_path, checkpoint_path
        )
        model = lightning_model.model

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Prithvi EO model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (pixel_values) that can be fed to the model.
        """
        # Create synthetic 6-band satellite imagery input (BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2)
        pixel_values = torch.randn(batch_size, 6, 512, 512)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return {"pixel_values": pixel_values}
