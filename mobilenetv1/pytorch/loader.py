# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MobilenetV1 model loader implementation
"""

from ...base import ForgeModel

from PIL import Image
from ...tools.utils import get_file
from torchvision import transforms
from .src.utils import MobileNetV1


class ModelLoader(ForgeModel):
    """Loads MobilenetV1 model and sample input."""

    # Shared configuration parameters
    model_name = "mobilenet_v1"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load pretrained MobilenetV1 model."""
        model = MobileNetV1(9)
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Prepare sample input for MobilenetV1 model"""

        # Get the Image
        image_file = get_file(
            "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
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
