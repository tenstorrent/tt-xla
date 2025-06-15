# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Segformer model loader implementation for question answering
"""
import torch
from transformers import (
    SegformerConfig,
    SegformerForImageClassification,
)
from PIL import Image

from transformers import AutoImageProcessor
from ...base import ForgeModel
from ...tools.utils import get_file


class ModelLoader(ForgeModel):
    # Shared configuration parameters
    model_name = "nvidia/mit-b0"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load a Segformer model from Hugging Face."""
        config = SegformerConfig.from_pretrained(cls.model_name)
        config_dict = config.to_dict()
        config_dict["return_dict"] = False
        config = SegformerConfig(**config_dict)
        model = SegformerForImageClassification.from_pretrained(
            cls.model_name, config=config
        )
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Generate sample inputs for Segformer models."""
        # Get the Image
        image = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")

        # Initialize tokenizer
        image_processor = AutoImageProcessor.from_pretrained(cls.model_name)

        # Create tokenized inputs
        inputs = image_processor(images=image, return_tensors="pt").pixel_values

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
