# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MGP-STR model loader implementation
"""

from ...base import ForgeModel
from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition
from PIL import Image
from ...tools.utils import get_file


class ModelLoader(ForgeModel):
    """Loads MGP-STR model and sample input."""

    # Shared configuration parameters
    model_name = "alibaba-damo/mgp-str-base"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load pretrained MGP-STR model."""

        model = MgpstrForSceneTextRecognition.from_pretrained(
            cls.model_name, return_dict=False
        )
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Prepare sample input for MGP-STR model"""

        # Get the Image
        image_file = get_file("https://i.postimg.cc/ZKwLg2Gw/367-14.png")
        image = Image.open(image_file).convert("RGB")

        # Preprocess image
        processor = MgpstrProcessor.from_pretrained(cls.model_name)
        inputs = processor(
            images=image,
            return_tensors="pt",
        ).pixel_values

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
