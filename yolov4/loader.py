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


class ModelLoader(ForgeModel):
    """YOLOv4 model loader implementation."""

    @classmethod
    def load_model(cls):
        """Load and return the YOLOv4 model instance with default settings.

        Returns:
            torch.nn.Module: The YOLOv4 model instance.
        """
        model = Yolov4()
        return model.to(torch.bfloat16)

    @classmethod
    def load_inputs(cls):
        """Load and return sample inputs for the YOLOv4 model with default settings.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """

        def url_to_image(url):
            resp = urllib.request.urlopen(url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return image

        img = url_to_image("http://images.cocodataset.org/val2017/000000039769.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (640, 480))  # Resize to model input size
        img = img / 255.0  # Normalize to [0,1]
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW format
        img = [torch.from_numpy(img).float().unsqueeze(0)]  # Add batch dimension
        batch_tensor = torch.cat(img, dim=0)

        return batch_tensor.to(torch.bfloat16)
