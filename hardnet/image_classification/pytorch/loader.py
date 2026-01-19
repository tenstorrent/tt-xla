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
from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ....base import ForgeModel
from ....tools.utils import print_compiled_model_results
from ....tools.utils import get_file


class ModelLoader(ForgeModel):
    """HardNet model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="hardnet",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the HardNet model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The HardNet model instance.
        """
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model_name = "hardnet68"
        model = torch.hub.load("PingoLH/Pytorch-HarDNet", model_name, pretrained=False)
        checkpoint = "https://github.com/PingoLH/Pytorch-HarDNet/raw/refs/heads/master/hardnet68.pth"
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                checkpoint, progress=False, map_location="cpu"
            )
        )
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the HardNet model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """

        file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
        input_image = Image.open(file_path)
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

    def print_cls_results(self, compiled_model_out):
        print_compiled_model_results(compiled_model_out)
