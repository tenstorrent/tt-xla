# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOX model loader implementation
"""
import torch
from PIL import Image
from torchvision import transforms

from ...base import ForgeModel
from ...tools.utils import get_file
import subprocess


subprocess.run(["pip", "install", "yolox==0.3.0", "--no-deps"])


class ModelLoader(ForgeModel):
    """YOLOX model loader implementation."""

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the YOLOX model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOX model instance.
        """
        variant = "yolox-tiny"
        from yolox.exp import get_exp

        exp = get_exp(
            exp_name=variant
        )  # yolox-tiny could be replaced by yolox-nano/s/m and so on
        model = exp.get_model()  # now you get yolox-tiny model
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Load and return sample inputs for the YOLOX model with default settings.

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
