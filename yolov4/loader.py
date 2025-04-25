# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv4 model loader implementation
"""
import torch
import cv2
import numpy as np
import urllib.request

from ..base import ForgeModel
from .src.yolov4 import Yolov4


class YOLOv4Loader(ForgeModel):
    """YOLOv4 model loader implementation."""

    @classmethod
    def load_model(cls, dtype=torch.bfloat16, **kwargs):
        """Load and return the YOLOv4 model instance.

        Args:
            dtype: The data type to convert the model to. Default is torch.bfloat16.
                  Set to None to keep the default model dtype.
            **kwargs: Additional arguments for model configuration.

        Returns:
            torch.nn.Module: The YOLOv4 model instance.
        """
        model = Yolov4()
        if dtype is not None:
            model = model.to(dtype)
        return model

    @classmethod
    def load_inputs(
        cls, dtype=torch.bfloat16, img_url=None, img_size=(640, 480), **kwargs
    ):
        """Load and return sample inputs for the YOLOv4 model.

        Args:
            dtype: The data type to convert the inputs to. Default is torch.bfloat16.
                 Set to None to keep the default tensor dtype (float32).
            img_url: URL of the image to load. Default is COCO dataset image.
            img_size: Size to resize the image to. Default is (640, 480).
            **kwargs: Additional arguments for input configuration.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """
        if img_url is None:
            img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

        # Load image from URL
        resp = urllib.request.urlopen(img_url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Preprocess the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, img_size)  # Resize to model input size
        img = img / 255.0  # Normalize to [0,1]
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW format
        img = [torch.from_numpy(img).float().unsqueeze(0)]  # Add batch dimension
        batch_tensor = torch.cat(img, dim=0)

        if dtype is not None:
            batch_tensor = batch_tensor.to(dtype)

        return batch_tensor
