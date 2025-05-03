# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOv3 model loader implementation
Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/yolo_v3/pytorch_yolov3_holli.py
"""
import torch
import requests
import os
from PIL import Image
from torchvision import transforms
from pathlib import Path

from ..base import ForgeModel
from .src.yolov3 import Yolov3


class ModelLoader(ForgeModel):
    """YOLOv3 model loader implementation."""

    @classmethod
    def load_model(cls):
        """Load and return the YOLOv3 model instance with default settings.

        Returns:
            torch.nn.Module: The YOLOv3 model instance.
        """
        num_classes = 80
        weights_url = (
            "https://www.ollihuotari.com/data/yolov3_pytorch/yolov3_coco_01.h5"
        )

        # Download model weights
        if (
            "DOCKER_CACHE_ROOT" in os.environ
            and Path(os.environ["DOCKER_CACHE_ROOT"]).exists()
        ):
            download_dir = Path(os.environ["DOCKER_CACHE_ROOT"]) / "custom_weights"
        else:
            download_dir = Path.home() / ".cache/custom_weights"
        download_dir.mkdir(parents=True, exist_ok=True)

        load_path = download_dir / weights_url.split("/")[-1]
        if not load_path.exists():
            response = requests.get(weights_url, stream=True)
            with open(str(load_path), "wb") as f:
                f.write(response.content)

        # Create model and load weights
        model = Yolov3(num_classes=num_classes)
        model.load_state_dict(
            torch.load(
                str(load_path),
                map_location=torch.device("cpu"),
            )
        )

        return model.to(torch.bfloat16)

    @classmethod
    def load_inputs(cls):
        """Load and return sample inputs for the YOLOv3 model with default settings.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """
        # Original image used in test
        image_url = (
            "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
        )

        # Download and load image
        image = Image.open(requests.get(image_url, stream=True).raw)

        # Preprocess the image
        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

        img_tensor = [transform(image).unsqueeze(0)]
        batch_tensor = torch.cat(img_tensor, dim=0)

        return batch_tensor.to(torch.bfloat16)
