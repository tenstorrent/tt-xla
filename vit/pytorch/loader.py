# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Vit model loader implementation for question answering
"""
import torch
from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image

from ...base import ForgeModel
from ...tools.utils import get_file


class ModelLoader(ForgeModel):
    # Shared configuration parameters
    model_name = "google/vit-large-patch16-224"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load a Vit model from Hugging Face."""
        model = ViTForImageClassification.from_pretrained(
            cls.model_name, return_dict=False
        )
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Generate sample inputs for Vit models."""
        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)
        # Initialize tokenizer
        image_processor = AutoImageProcessor.from_pretrained(
            cls.model_name, use_fast=True
        )

        # Create tokenized inputs
        inputs = image_processor(images=image, return_tensors="pt").pixel_values

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
