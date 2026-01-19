# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLPN-KITTI model loader implementation
"""
import torch


from PIL import Image
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ....base import ForgeModel
from ....tools.utils import get_file


class ModelLoader(ForgeModel):
    """GLPN-KITTI model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "vinvino02/glpn-kitti"
        self.processor = None
        self.image = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="glpn_kitti",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the GLPN-KITTI model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The GLPN-KITTI model instance.
        """
        self.processor = GLPNImageProcessor.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = GLPNForDepthEstimation.from_pretrained(self.model_name, **model_kwargs)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the GLPN-KITTI model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        self.image = Image.open(str(image_file))

        # Ensure processor is initialized
        if self.processor is None:
            self.load_model(dtype_override=dtype_override)
        # prepare image for the model
        inputs = self.processor(images=self.image, return_tensors="pt")

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        # Replicate inputs for batch size
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
            elif isinstance(inputs[key], list):
                inputs[key] = inputs[key] * batch_size

        return inputs
