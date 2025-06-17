# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/linear_autoencoder/pytorch_linear_autoencoder.py
"""
Deit model loader implementation
"""


from ...base import ForgeModel
from PIL import Image
from transformers import AutoFeatureExtractor, ViTForImageClassification
from ...tools.utils import get_file


class ModelLoader(ForgeModel):
    """Deit model loader implementation."""

    # Shared configuration parameters
    model_name = "facebook/deit-base-patch16-224"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the Deit model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Deit model instance.
        """
        # Initialize feature extractor first
        cls.feature_extractor = AutoFeatureExtractor.from_pretrained(cls.model_name)

        # Load pre-trained model from HuggingFace
        model = ViTForImageClassification.from_pretrained(cls.model_name)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Deit model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Preprocessed input tensors suitable for ViTForImageClassification.
        """

        # Ensure feature extractor is initialized
        if not hasattr(cls, "feature_extractor"):
            cls.load_model(
                dtype_override=dtype_override
            )  # This will initialize the feature extractor

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_file))
        batch_images = [image] * batch_size  # Create a batch of images
        inputs = cls.feature_extractor(images=batch_images, return_tensors="pt")

        return inputs
