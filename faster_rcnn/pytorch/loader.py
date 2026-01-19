# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Faster R-CNN model loader implementation for object detection.
"""
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms as T
from typing import Optional
from PIL import Image

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available Faster R-CNN model variants for object detection."""

    RESNET50_FPN = "resnet50_fpn"


class ModelLoader(ForgeModel):
    """Faster R-CNN model loader implementation for object detection tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.RESNET50_FPN: ModelConfig(
            pretrained_model_name="torchvision/fasterrcnn_resnet50_fpn",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.RESNET50_FPN

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="faster_rcnn",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.TORCHVISION,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Faster R-CNN model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           NOTE: This parameter is currently ignored (model always uses float32).

        Returns:
            torch.nn.Module: The Faster R-CNN model instance for object detection.
        """
        # Load pretrained Faster R-CNN model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        model.eval()

        # NOTE: Ignoring dtype_override and always using default (fp32)
        # because "nms_kernel" not implemented for 'BFloat16'
        # if dtype_override is not None:
        #     model = model.to(dtype=dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Faster R-CNN model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
                           NOTE: This parameter is currently ignored (model always uses float32).
            batch_size: Batch size for the inputs.

        Returns:
            list[list[torch.Tensor]]: input tensors that can be fed to the model.
        """
        # Download and load image
        img_path = get_file(
            "https://cdn.pixabay.com/photo/2013/07/05/01/08/traffic-143391_960_720.jpg"
        )
        img_pil = Image.open(img_path).convert("RGB")

        # Define and apply transform
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(img_pil)

        # NOTE: Ignoring dtype_override and always using default (fp32)
        # because "nms_kernel" not implemented for 'BFloat16'
        # if dtype_override is not None:
        #     img_tensor = img_tensor.to(dtype_override)

        # Create a list of images based on batch size
        inputs = [img_tensor for _ in range(batch_size)]

        return [inputs]
