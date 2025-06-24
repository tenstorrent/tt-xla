# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLPN-KITTI model loader implementation
"""
import torch


from PIL import Image
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
from ...base import ForgeModel
from ...tools.utils import get_file


class ModelLoader(ForgeModel):
    """GLPN-KITTI model loader implementation."""

    # Shared configuration parameters
    model_name = "vinvino02/glpn-kitti"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the GLPN-KITTI model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The GLPN-KITTI model instance.
        """
        cls.processor = GLPNImageProcessor.from_pretrained(cls.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = GLPNForDepthEstimation.from_pretrained(cls.model_name, **model_kwargs)
        return model

    @classmethod
    def load_inputs(cls, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the GLPN-KITTI model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        cls.image = Image.open(str(image_file))

        # Ensure processor is initialized
        if not hasattr(cls, "processor"):
            cls.load_model(dtype_override=dtype_override)
        # prepare image for the model
        inputs = cls.processor(images=cls.image, return_tensors="pt")

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        # Replicate inputs for batch size
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
            elif isinstance(inputs[key], list):
                inputs[key] = inputs[key] * batch_size

        return inputs
