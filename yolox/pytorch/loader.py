# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOX model loader implementation
"""
import subprocess

subprocess.run(
    ["pip", "install", "yolox==0.3.0", "--no-deps"]
)  # Install yolox==0.3.0 without installing its dependencies

import torch
import cv2
import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from yolox.data.data_augment import preproc as preprocess

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel
from ...tools.utils import get_file
from .src.utils import print_detection_results
import subprocess


class ModelLoader(ForgeModel):
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
            model="yolox",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    """YOLOX model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_variant = "yolox-tiny"
        self.input_shape = (416, 416)

    def load_model(self, dtype_override=None):
        """Load and return the YOLOX model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOX model instance.
        """
        from yolox.exp import get_exp

        model_name = self.model_variant.replace("-", "_")
        weight_url = f"https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{model_name}.pth"

        # Use the utility to download/cache the model weights
        weight_path = get_file(weight_url)

        # Load model
        exp = get_exp(exp_name=model_name)
        model = exp.get_model()
        ckpt = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the YOLOX model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """
        image_path = get_file("http://images.cocodataset.org/val2017/000000397133.jpg")
        img = cv2.imread(str(image_path))
        img_tensor, ratio = preprocess(img, self.input_shape)
        self.ratio = ratio
        img_tensor = torch.from_numpy(img_tensor)
        batch_tensor = img_tensor.unsqueeze(0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor

    def post_processing(self, co_out):
        print_detection_results(co_out, self.ratio, self.input_shape)
