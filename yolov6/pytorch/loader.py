# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv6 model loader implementation
"""

from typing import Optional

from ...tools.utils import get_file
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
from .src.utils import check_img_size, process_image


class ModelVariant(StrEnum):
    """Available YOLOv6 model variants."""

    YOLOV6N = "yolov6n"
    YOLOV6S = "yolov6s"
    YOLOV6M = "yolov6m"
    YOLOV6L = "yolov6l"


class ModelLoader(ForgeModel):
    """YOLOv6 model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.YOLOV6N: ModelConfig(
            pretrained_model_name="yolov6n",
        ),
        ModelVariant.YOLOV6S: ModelConfig(
            pretrained_model_name="yolov6s",
        ),
        ModelVariant.YOLOV6M: ModelConfig(
            pretrained_model_name="yolov6m",
        ),
        ModelVariant.YOLOV6L: ModelConfig(
            pretrained_model_name="yolov6l",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.YOLOV6S

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
        return ModelInfo(
            model="yolov6",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):

        from yolov6.layers.common import DetectBackend

        """Load and return the YOLOv6 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOv6 model instance.
        """
        # Get the model name from the instance's variant config
        variant = self._variant_config.pretrained_model_name
        weight_url = (
            f"https://github.com/meituan/YOLOv6/releases/download/0.3.0/{variant}.pt"
        )

        # Use the utility to download/cache the model weights
        weight_path = get_file(weight_url)

        model = DetectBackend(weight_path)
        framework_model = model.model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            framework_model = framework_model.to(dtype_override)

        return framework_model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the YOLOv6 model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """

        stride = 32
        input_size = 640
        img_size = check_img_size(input_size, s=stride)
        img, img_src = process_image(img_size, stride, half=False)
        input_batch = img.unsqueeze(0)

        # Replicate tensors for batch size
        batch_tensor = input_batch.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor
