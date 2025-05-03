# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OFT (Orthographic Feature Transform) model loader implementation
"""
import torch

from ..base import ForgeModel
from .src.oftnet import OftNet


class ModelLoader(ForgeModel):
    """OFT model loader implementation."""

    # Shared configuration parameters
    grid_res = 0.5
    num_classes = 1
    frontend = "resnet18"
    topdown_layers = 8
    grid_height = 4.0

    @classmethod
    def load_model(cls):
        """Load and return the OFT model instance with default settings.

        Returns:
            torch.nn.Module: The OFT model instance.
        """
        # Load model with defaults
        model = OftNet(
            num_classes=cls.num_classes,
            frontend=cls.frontend,
            topdown_layers=cls.topdown_layers,
            grid_res=cls.grid_res,
            grid_height=cls.grid_height,
        )

        return model

    @classmethod
    def load_inputs(cls):
        """Load and return sample inputs for the OFT model with default settings.

        Returns:
            tuple: Sample inputs that can be fed to the model.
                - dummy_image (torch.Tensor): Input image tensor [B, 3, H, W]
                - dummy_calib (torch.Tensor): Camera calibration parameters [B, 3, 4]
                - dummy_grid (torch.Tensor): Bird's-eye view grid [B, D, W, 3]
        """
        # Fixed default parameters
        batch_size = 1
        grid_size = (80.0, 80.0)  # width, depth in meters

        # Create input tensors matching the original test
        dummy_image = torch.randn(batch_size, 3, 224, 224)
        dummy_calib = torch.randn(batch_size, 3, 4)

        # Calculate grid dimensions
        grid_depth = int(grid_size[1] / cls.grid_res)
        grid_width = int(grid_size[0] / cls.grid_res)
        dummy_grid = torch.randn(batch_size, grid_depth, grid_width, 3)

        return (dummy_image, dummy_calib, dummy_grid)
