# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HardNet model loader implementation
"""
import torch


from PIL import Image
from torchvision import transforms
import requests
import torch
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """HardNet model loader implementation."""

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the HardNet model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The HardNet model instance.
        """
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load("PingoLH/Pytorch-HarDNet", "hardnet68", pretrained=False)
        checkpoint = "https://github.com/PingoLH/Pytorch-HarDNet/raw/refs/heads/master/hardnet68.pth"
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                checkpoint, progress=False, map_location="cpu"
            )
        )
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the HardNet model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """

        url = "https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png"
        input_image = Image.open(requests.get(url, stream=True).raw)
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = torch.stack(
            [input_tensor] * batch_size, dim=0
        )  # create a mini-batch as expected by the model

        if dtype_override is not None:
            input_batch = input_batch.to(dtype_override)

        return input_batch
