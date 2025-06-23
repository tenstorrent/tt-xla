# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv8 model loader implementation
"""
import torch
import cv2
import numpy as np
from ...tools.utils import get_file
from ...base import ForgeModel
from torch.hub import load_state_dict_from_url
from ultralytics.nn.tasks import DetectionModel


class ModelLoader(ForgeModel):
    """YOLOv8 model loader implementation."""

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the YOLOv8 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOv8 model instance.
        """

        variant = "yolov8x"
        weights = load_state_dict_from_url(
            f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{variant}.pt"
        )
        model = DetectionModel(cfg=weights["model"].yaml)
        model.load_state_dict(weights["model"].float().state_dict())
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Load and return sample inputs for the YOLOv8 model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        img = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (640, 480))  # Resize to model input size
        img = img / 255.0  # Normalize to [0,1]
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW format
        img = [torch.from_numpy(img).float().unsqueeze(0)]  # Add batch dimension
        batch_tensor = torch.cat(img, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor
