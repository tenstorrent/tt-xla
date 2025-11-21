# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv9 model loader implementation
"""

from typing import Optional
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available YOLOv9 model variants."""

    T = "t"
    S = "s"
    M = "m"
    C = "c"
    E = "e"


class ModelLoader(ForgeModel):
    """YOLOv9 model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.T: ModelConfig(
            pretrained_model_name="yolov9-t-converted",
        ),
        ModelVariant.S: ModelConfig(
            pretrained_model_name="yolov9-s-converted",
        ),
        ModelVariant.M: ModelConfig(
            pretrained_model_name="yolov9-m-converted",
        ),
        ModelVariant.C: ModelConfig(
            pretrained_model_name="yolov9-c-converted",
        ),
        ModelVariant.E: ModelConfig(
            pretrained_model_name="yolov9-e-converted",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.T

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        if variant == ModelVariant.T:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="YOLOv9",
            variant=variant,
            group=group,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the YOLOv9 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOv9 model instance.
        """
        from .src.model_utils import attempt_load

        # Construct weights URL dynamically from variant
        weights_url = f"https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-{self._variant}-converted.pt"
        weight_path = get_file(weights_url)

        # Load model
        model = attempt_load(weight_path, "cpu", inplace=True, fuse=True)
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        # Store model for later use in load_inputs
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return inputs for the YOLOv9 model

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for YOLOv9.
        """
        import torch
        import cv2
        import numpy as np
        from .src.model_utils import letterbox, check_img_size

        # Get sample image from COCO dataset
        image_file = get_file("test_images/horses.jpg")

        # Get model stride and calculate proper image size
        stride = int(self.model.stride.max())
        img_size = check_img_size(640, s=stride)

        # Load and preprocess image
        im0 = cv2.imread(str(image_file))
        im = letterbox(im0, img_size, stride=stride, auto=True)[0]

        # Convert to tensor format (HWC to CHW, BGR to RGB)
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im)
        im = im.float()
        im /= 255.0

        # Add batch dimension
        if len(im.shape) == 3:
            im = im[None]

        # Replicate tensors for batch size
        im = im.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            im = im.to(dtype_override)

        return im
