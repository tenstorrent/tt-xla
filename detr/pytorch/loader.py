# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DETR model loader implementation
"""
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from ...tools.utils import get_file
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """DETR model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "detr_resnet50"
        self.model_src = "facebookresearch/detr:main"

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
            model="DETR",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the DETR model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DETR model instance.

        The model is from https://github.com/facebookresearch/detr
        """
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load(self.model_src, self.model_name, pretrained=True)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DETR model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        image_file = get_file(
            "https://huggingface.co/spaces/nakamura196/yolov5-char/resolve/8a166e0aa4c9f62a364dafa7df63f2a33cbb3069/ultralytics/yolov5/data/images/zidane.jpg"
        )
        input_image = Image.open(str(image_file))
        m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=m, std=s),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = torch.stack(
            [input_tensor] * batch_size
        )  # Create batch of size `batch_size`
        if dtype_override is not None:
            input_batch = input_batch.to(dtype_override)
        return input_batch
