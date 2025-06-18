# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Regnet model loader implementation
"""

from ...base import ForgeModel
from transformers import AutoFeatureExtractor, RegNetForImageClassification
from PIL import Image
from ...tools.utils import get_file


class ModelLoader(ForgeModel):
    """Loads Regnet model and sample input."""

    # Shared configuration parameters
    model_name = "facebook/regnet-y-040"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load pretrained Regnet model."""

        model = RegNetForImageClassification.from_pretrained(
            cls.model_name, return_dict=False
        )
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Prepare sample input for Regnet model"""

        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)

        # Preprocess image
        image_processor = AutoFeatureExtractor.from_pretrained(cls.model_name)
        inputs = image_processor(images=image, return_tensors="pt").pixel_values

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
