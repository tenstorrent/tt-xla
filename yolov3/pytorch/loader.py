# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOv3 model loader implementation
Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/yolo_v3/pytorch_yolov3_holli.py
"""
import torch
from torchvision import transforms
from typing import Optional
from PIL import Image
from datasets import load_dataset

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
from ...tools.utils import VisionPreprocessor, get_file


class ModelVariant(StrEnum):
    """Available YOLOv3 model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """YOLOv3 model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="test_files/pytorch/yolov3/yolov3_coco_01.h5",
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
        self.model = None
        self._preprocessor = None

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

        # Store model for potential use in input preprocessing
        self.model = model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def input_preprocess(self, dtype_override=None, batch_size=1, image=None):
        """Preprocess input image(s) and return model-ready input tensor.

        Args:
            dtype_override: Optional torch.dtype override (default: float32).
            batch_size: Batch size (ignored if image is a list).
            image: PIL Image, URL string, tensor, list of images/URLs, or None (uses default dataset image).

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        if self._preprocessor is None:
            # YOLOv3 uses custom preprocessing: resize to 512x512 and ToTensor
            def custom_preprocess_fn(img: Image.Image) -> torch.Tensor:
                transform = transforms.Compose(
                    [
                        transforms.Resize((512, 512)),
                        transforms.ToTensor(),
                    ]
                )
                return transform(img)

            self._preprocessor = VisionPreprocessor(
                model_source=ModelSource.CUSTOM,
                model_name="yolov3",
                custom_preprocess_fn=custom_preprocess_fn,
            )

        # If image is None, use default dataset image (backward compatibility)
        if image is None:
            dataset = load_dataset("chrismontes/dog_images", split="train")
            image = dataset[0]["image"]

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs (backward compatibility wrapper for input_preprocess).

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).
            image: Optional input image.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        return self.input_preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )
