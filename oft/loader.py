# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OFT (Orthographic Feature Transform) model loader implementation
"""
import torch

from ..base import ForgeModel
from .src.oftnet import OftNet


class OFTLoader(ForgeModel):
    """OFT model loader implementation."""

    @classmethod
    def load_model(
        cls,
        dtype=None,
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=0.5,
        grid_height=4.0,
        **kwargs
    ):
        """Load and return the OFT model instance.

        Args:
            dtype: The data type to convert the model to. Default is None (keep original).
            num_classes: Number of classes to predict.
            frontend: Frontend feature extractor to use, default is 'resnet18'.
            topdown_layers: Number of layers in the topdown network.
            grid_res: Grid resolution for the BEV representation.
            grid_height: Height of the BEV grid in meters.
            **kwargs: Additional arguments for model configuration.

        Returns:
            torch.nn.Module: The OFT model instance.
        """
        model = OftNet(
            num_classes=num_classes,
            frontend=frontend,
            topdown_layers=topdown_layers,
            grid_res=grid_res,
            grid_height=grid_height,
            **kwargs
        )

        if dtype is not None:
            model = model.to(dtype)

        return model

    @classmethod
    def load_inputs(
        cls, dtype=None, batch_size=1, grid_res=0.5, grid_size=(80.0, 80.0), **kwargs
    ):
        """Load and return sample inputs for the OFT model.

        Args:
            dtype: The data type to convert the inputs to. Default is None (keep as float32).
            batch_size: Batch size for the input tensors.
            grid_res: Grid resolution for the BEV representation.
            grid_size: Size of the grid as (width, depth) in meters.
            **kwargs: Additional arguments for input configuration.

        Returns:
            tuple: A tuple containing (image, calibration, grid) tensors.
        """
        # Create dummy inputs for the model
        dummy_image = torch.randn(batch_size, 3, 224, 224)
        dummy_calib = torch.randn(batch_size, 3, 4)

        # Calculate grid dimensions
        grid_depth = int(grid_size[1] / grid_res)
        grid_width = int(grid_size[0] / grid_res)
        dummy_grid = torch.randn(batch_size, grid_depth, grid_width, 3)

        # Convert to requested dtype if specified
        if dtype is not None:
            dummy_image = dummy_image.to(dtype)
            dummy_calib = dummy_calib.to(dtype)
            dummy_grid = dummy_grid.to(dtype)

        return (dummy_image, dummy_calib, dummy_grid)
