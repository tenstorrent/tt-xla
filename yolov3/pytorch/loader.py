# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOv3 model loader implementation
Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/yolo_v3/pytorch_yolov3_holli.py
"""
import torch
from PIL import Image
from torchvision import transforms
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
from .src.yolov3 import Yolov3
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available YOLOv3 model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """YOLOv3 model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="https://www.ollihuotari.com/data/yolov3_pytorch/yolov3_coco_01.h5",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="yolov3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    def load_model(self, dtype_override=None):
        """Load and return the YOLOv3 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOv3 model instance.
        """
        num_classes = 80
        weights_file = get_file(self._variant_config.pretrained_model_name)

        # Create model and load weights
        model = Yolov3(num_classes=num_classes)
        model.load_state_dict(
            torch.load(
                str(weights_file),
                map_location=torch.device("cpu"),
            )
        )

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the YOLOv3 model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Number of samples in the batch. Default is 1.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """
        # Original image used in test
        image_file = get_file(
            "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
        )

        # Download and load image
        image = Image.open(image_file)

        # Preprocess the image
        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

        batch_tensor = torch.stack([transform(image)] * batch_size)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor
