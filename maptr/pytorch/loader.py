# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MAPTR model loader implementation
"""
import torch
from typing import Optional
from mmcv.runner import load_checkpoint

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
from ...tools.utils import get_file
from .src.maptr import build_model, bevformer_cfg
from mmcv import ConfigDict
import numpy as np


class ModelVariant(StrEnum):
    """Available MAPTR model variants."""

    TINY_R50_24E_BEVFORMER = "tiny_r50_24e_bevformer"
    TINY_R50_24E_BEVFORMER_T4 = "tiny_r50_24e_bevformer_t4"


class ModelLoader(ForgeModel):
    """MAPTR model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.TINY_R50_24E_BEVFORMER: ModelConfig(
            pretrained_model_name="tiny_r50_24e_bevformer",
        ),
        ModelVariant.TINY_R50_24E_BEVFORMER_T4: ModelConfig(
            pretrained_model_name="tiny_r50_24e_bevformer_t4",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.TINY_R50_24E_BEVFORMER

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant.
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="maptr",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.REALTIME_MAP_CONSTRUCTION,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self):
        """Load pretrained MAPTR model for this instance's variant.

        Returns:
            torch.nn.Module: The MAPTR model instance.
        """
        # Get the variant name from the instance's variant config
        variant_name = self._variant_config.pretrained_model_name

        # Get pretrained weights
        model_path = str(get_file(f"test_files/pytorch/maptr/maptr_{variant_name}.pth"))

        if variant_name == "tiny_r50_24e_bevformer_t4":
            # Update model config
            bevformer_cfg["pts_bbox_head"]["transformer"]["len_can_bus"] = 6

        # wrap the model config
        self.cfg = ConfigDict(bevformer_cfg)

        # Build model
        model = build_model(self.cfg)

        # Load pretrained weights
        load_checkpoint(model, model_path, map_location="cpu")
        model.eval()

        return model

    def load_inputs(self):
        """Prepare dummy sample inputs for the MAPTR model (NuScenes format).

        This function creates synthetic test data mimicking the structure expected by
        MAPTR when using the NuScenes dataset. It does not load real data, but instead
        generates placeholder inputs for testing purposes:

        - img_shape: List of image shapes, one per camera view.
        - lidar2img: List of 4x4 projection matrices (identity matrices here as placeholders).
        - can_bus: Vehicle state vector (zeros used as dummy values).
        - img: Random image tensor simulating 6 camera views.

        Returns:
            dict: Dictionary containing:
                - ``img_metas`` (list): Metadata with image shapes, projection matrices,
                CAN bus vector, and a dummy scene token.
                - ``img`` (list): Randomized image tensor of shape (1, 6, 3, 480, 800).
        """

        img_shape = [(480, 800, 3)] * 6
        lidar2img = [np.eye(4) for _ in range(6)]
        can_bus = np.zeros(18, dtype=np.float64)
        img = torch.randn(1, 6, 3, 480, 800)

        data = {
            "img_metas": [
                [
                    {
                        "img_shape": img_shape,
                        "lidar2img": lidar2img,
                        "can_bus": can_bus,
                        "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
                    }
                ]
            ],
            "img": [img],
        }

        return data
