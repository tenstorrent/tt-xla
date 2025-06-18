# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Vgg model loader implementation
"""

import torch
from ...base import ForgeModel

from PIL import Image
from ...tools.utils import get_file
from torchvision import transforms


class ModelLoader(ForgeModel):
    """Loads Vgg model and sample input."""

    # Shared configuration parameters
    model_name = "vgg19_bn"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load pretrained Vgg model."""

        model = torch.hub.load(
            "pytorch/vision:v0.10.0", cls.model_name, pretrained=True
        )
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Prepare sample input for Vgg model"""

        # Get the Image
        image_file = get_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        )
        image = Image.open(image_file)

        # Preprocess image
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
        inputs = preprocess(image).unsqueeze(0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
