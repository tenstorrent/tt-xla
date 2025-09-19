# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BEVFormer model loader implementation
"""
import torch
import numpy as np
from typing import Optional
from ...base import ForgeModel
from ...config import ModelGroup, ModelTask, ModelSource, Framework, StrEnum, ModelInfo
from .src.model import (
    BEVFormer,
    img_backbone,
    pts_bbox_head,
    img_neck,
    load_checkpoint_bev,
)
from .src.model_utils import generate_random_lidar2img, generate_random_can_bus
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available BEVFormer model variants."""

    BEVFORMER_TINY = "BEVFormer-tiny"


class ModelLoader(ForgeModel):
    """BEVFormer model loader implementation for autonomous driving tasks."""

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BEVFORMER_TINY

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        # Configuration parameters
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="bevformer",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the BEVFormer model instance with default settings.
        Returns:
            Torch model: The BEVFormer model instance.
        """
        # Load model with defaults
        model = BEVFormer(
            img_backbone=img_backbone,
            pts_bbox_head=pts_bbox_head,
            img_neck=img_neck,
            use_grid_mask=True,
            video_test_mode=True,
        )
        checkpoint_path = str(
            get_file("test_files/pytorch/bevformer/bevformer_tiny_epoch_24.pth")
        )
        checkpoint = load_checkpoint_bev(
            model,
            checkpoint_path,
            map_location="cpu",
        )
        model.eval()
        return model

    def load_inputs(self, **kwargs):
        """Return sample inputs for the BEVFormer model with default settings.
        Returns:
            dict: A dictionary of input tensors and metadata suitable for the model.
        """

        img_shapes = [
            (480, 800, 3),
            (480, 800, 3),
            (480, 800, 3),
            (480, 800, 3),
            (480, 800, 3),
            (480, 800, 3),
        ]
        lidar2img = generate_random_lidar2img()

        can_bus = generate_random_can_bus()
        img = torch.randn(1, 6, 3, 480, 800)
        img_metas = {
            "img_shape": img_shapes,
            "lidar2img": lidar2img,
            "scale_factor": 1.0,
            "flip": False,
            "pcd_horizontal_flip": False,
            "pcd_vertical_flip": False,
            "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
            "prev_idx": "",
            "next_idx": "3950bd41f74548429c0f7700ff3d8269",
            "pcd_scale_factor": 1.0,
            "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
            "can_bus": can_bus,
        }
        input_dict = {"rescale": True, "img_metas": [[img_metas]], "img": [img]}
        return input_dict
