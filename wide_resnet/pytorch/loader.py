# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WideResnet model loader implementation for question answering
"""
import torch
from PIL import Image
from torchvision import transforms
from ...base import ForgeModel
from ...tools.utils import get_file


class ModelLoader(ForgeModel):
    # Shared configuration parameters
    model_name = "wide_resnet50_2"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load a WideResnet model from Torch Hub."""
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
        """Generate sample inputs for WideResnet models."""
        # Get the Image
        image_file = get_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        )
        input_image = Image.open(image_file)

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
        inputs = input_tensor.unsqueeze(0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
