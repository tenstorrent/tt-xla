# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/linear_autoencoder/pytorch_linear_autoencoder.py
"""
Clip model loader implementation
"""


from ...base import ForgeModel
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ...tools.utils import get_file


class ModelLoader(ForgeModel):
    """CLIP model loader implementation."""

    # Shared configuration parameters
    model_name = "openai/clip-vit-base-patch32"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the CLIP model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The CLIP model instance.
        """
        # Initialize processor first with default or overridden dtype
        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["torch_dtype"] = dtype_override

        cls.processor = CLIPProcessor.from_pretrained(
            cls.model_name, **processor_kwargs
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = CLIPModel.from_pretrained(cls.model_name, **model_kwargs)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the CLIP model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors, pixel values and attention masks that can be fed to the model.
        """

        # Ensure processor is initialized
        if not hasattr(cls, "processor"):
            cls.load_model()  # This will initialize the processor

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_file))
        batch_images = [image] * batch_size  # Create a batch of images

        text = ["a photo of a cat", "a photo of a dog"]
        batch_text = [text] * batch_size  # Create a batch of text

        inputs = cls.processor(
            text=batch_text,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
