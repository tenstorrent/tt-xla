# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ViLT model loader implementation
"""

from transformers import ViltForQuestionAnswering, ViltProcessor
from ...base import ForgeModel
from PIL import Image
from ...tools.utils import get_file


class ModelLoader(ForgeModel):

    # Shared configuration parameters
    model_name = "dandelin/vilt-b32-finetuned-vqa"
    text = "How many cats are there?"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load a ViLT model from Hugging Face."""

        # Initialize processor first with default or overridden dtype
        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["torch_dtype"] = dtype_override

        cls.processor = ViltProcessor.from_pretrained(
            cls.model_name, **processor_kwargs
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = ViltForQuestionAnswering.from_pretrained(
            cls.model_name, return_dict=False, **model_kwargs
        )
        model.eval()
        return model

    @classmethod
    def load_inputs(cls):
        """Generate sample inputs for ViLT model."""

        # Ensure processor is initialized
        if not hasattr(cls, "processor"):
            cls.load_model()  # This will initialize the processor

        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)
        inputs = cls.processor(image, cls.text, return_tensors="pt")

        return inputs
