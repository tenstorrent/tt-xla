# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MLP-Mixer model loader implementation
"""

from ...base import ForgeModel
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
from ...tools.utils import get_file


class ModelLoader(ForgeModel):
    """Loads MLP-Mixer model and sample input."""

    # Shared configuration parameters
    model_name = "mixer_s32_224"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load pretrained MLP-Mixer model."""

        # To avoid RuntimeError: No pretrained weights exist for mixer_s32_224. Use `pretrained=False` for random init.
        model = timm.create_model(cls.model_name, pretrained=False)
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Prepare sample input for MLP-Mixer model"""

        # Get the Image
        image_file = get_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        )
        image = Image.open(image_file)

        # Preprocess image
        data_config = resolve_data_config({}, model=cls.load_model())
        transforms = create_transform(**data_config)
        inputs = transforms(image).unsqueeze(0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
