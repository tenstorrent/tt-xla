# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOv3 model loader implementation
Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/yolo_v3/pytorch_yolov3_holli.py
"""
import torch
from PIL import Image
from torchvision import transforms

from ...base import ForgeModel
from .src.yolov3 import Yolov3
from ...tools.utils import get_file


class ModelLoader(ForgeModel):
    """YOLOv3 model loader implementation."""

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the YOLOv3 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOv3 model instance.
        """
        num_classes = 80
        weights_file = get_file(
            "https://www.ollihuotari.com/data/yolov3_pytorch/yolov3_coco_01.h5"
        )

        # Create model and load weights
        model = Yolov3(num_classes=num_classes)
        model.load_state_dict(
            torch.load(
                str(weights_file),
                map_location=torch.device("cpu"),
            )
        )

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Load and return sample inputs for the YOLOv3 model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """
        # Original image used in test
        image_file = get_file(
            "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
        )

        # Download and load image
        image = Image.open(image_file)

        # Preprocess the image
        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

        img_tensor = [transform(image).unsqueeze(0)]
        batch_tensor = torch.cat(img_tensor, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor
