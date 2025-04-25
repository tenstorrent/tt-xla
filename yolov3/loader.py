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


class YOLOv3Loader(ForgeModel):
    """YOLOv3 model loader implementation."""

    @classmethod
    def load_model(
        cls, dtype=torch.bfloat16, num_classes=80, weights_url=None, **kwargs
    ):
        """Load and return the YOLOv3 model instance.

        Args:
            dtype: The data type to convert the model to. Default is torch.bfloat16.
                  Set to None to keep the default model dtype.
            num_classes: Number of classes to use in the model.
            weights_url: URL to download weights from. If None, uses default weights.
            **kwargs: Additional arguments for model configuration.

        Returns:
            torch.nn.Module: The YOLOv3 model instance.
        """
        # Use default URL if not provided
        if weights_url is None:
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

        # Load model
        model = Yolov3(num_classes=num_classes)
        model.load_state_dict(
            torch.load(
                str(load_path),
                map_location=torch.device("cpu"),
            )
        )

        if dtype is not None:
            model = model.to(dtype)

        return model

    @classmethod
    def load_inputs(
        cls, dtype=torch.bfloat16, image_url=None, img_size=(512, 512), **kwargs
    ):
        """Load and return sample inputs for the YOLOv3 model.

        Args:
            dtype: The data type to convert the inputs to. Default is torch.bfloat16.
                 Set to None to keep the default tensor dtype (float32).
            image_url: URL of the image to load. Default is a dog image.
            img_size: Size to resize the image to. Default is (512, 512).
            **kwargs: Additional arguments for input configuration.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """
        # Use default URL if not provided
        if image_url is None:
            image_url = (
                "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
            )

        # Load and preprocess image
        image = Image.open(requests.get(image_url, stream=True).raw)
        transform = transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor()]
        )
        img_tensor = [transform(image).unsqueeze(0)]
        batch_tensor = torch.cat(img_tensor, dim=0)

        if dtype is not None:
            batch_tensor = batch_tensor.to(dtype)

        return batch_tensor
