# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CenterNet model loader implementation
"""
import torch
import onnx
from PIL import Image
from torchvision import transforms
import numpy as np

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ....base import ForgeModel
from ....config import ModelGroup, ModelTask, ModelSource, Framework
from ....tools.utils import get_file


class ModelLoader(ForgeModel):
    """CenterNet model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'dla1x_od'.
                The task is determined based on the variant name:
                - If 'hpe' in variant name: pose estimation
                - If '3d' in variant name: 3D detection
                - Else: object detection

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "dla1x_od"  # Default variant is object detection
        # Determine task based on variant name
        if "hpe" in variant_name:
            task = ModelTask.CV_KEYPOINT_DET
        elif "3d" in variant_name.lower():
            task = ModelTask.CV_OBJECT_DET
        else:
            task = ModelTask.CV_OBJECT_DET
        return ModelInfo(
            model="centernet",
            variant=variant_name,
            group=ModelGroup.RED,
            task=task,
            source=ModelSource.CUSTOM,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the CenterNet model instance with default settings.

        Returns:
            Onnx model: The CenterNet model instance.
        """
        # Load model with defaults
        variant_name = kwargs.get("variant_name", "dla1x_od")
        path = f"test_files/onnx/centernet/{variant_name}.onnx"
        file = get_file(path)
        model = onnx.load(file)

        return model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the CenterNet model with default settings.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """
        # Create a random input tensor with the correct shape, using default dtype
        variant_name = kwargs.get("variant_name", "dla1x_od")
        # Set input resolution based on the task:
        # Use 512x512 for Object Detection (OD) and Human Pose Estimation (HPE) variants,
        # and 1280x384 for 3D Bounding Box (3D_BB) Detection variants
        if "od" in variant_name or "hpe" in variant_name:
            h, w = 512, 512
        else:
            h, w = 1280, 384
        image_file = get_file(
            "https://github.com/xingyizhou/CenterNet/raw/master/images/17790319373_bd19b24cfc_k.jpg"
        )
        image = Image.open(image_file).convert("RGB").resize((h, w))
        m, s = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))
        preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=m, std=s)]
        )
        input_tensor = preprocess(image)
        inputs = input_tensor.unsqueeze(0)

        return inputs
