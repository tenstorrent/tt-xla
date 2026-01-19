# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv6 model loader implementation
"""

from typing import Optional
import torch
import cv2
import numpy as np
from PIL import Image

from ....tools.utils import get_file
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from .src.utils import check_img_size, letterbox
from yolov6.core.inferer import Inferer
from yolov6.utils.nms import non_max_suppression
import yaml


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
        self.model = None
        self.img_src = None
        self.input_batch = None

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
        """Load and return the YOLOv6 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOv6 model instance.
        """
        # Disable weights_only check for PyTorch 2.6+ to load YOLOv6 checkpoint with custom classes
        import os

        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

        from yolov6.layers.common import DetectBackend

        # Get the model name from the instance's variant config
        variant = self._variant_config.pretrained_model_name
        weight_url = (
            f"https://github.com/meituan/YOLOv6/releases/download/0.3.0/{variant}.pt"
        )

        # Use the utility to download/cache the model weights
        weight_path = get_file(weight_url)

        model = DetectBackend(weight_path)
        framework_model = model.model
        framework_model.eval()

        # Store model for potential use
        self.model = framework_model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            framework_model = framework_model.to(dtype_override)
            self.model = framework_model

        return framework_model

    def input_preprocess(self, dtype_override=None, batch_size=1, image=None):
        """Preprocess input image(s) and return model-ready input tensor.

        Args:
            dtype_override: Optional torch.dtype override (default: float32).
            batch_size: Batch size (ignored if image is a list).
            image: PIL Image, URL string, numpy array (BGR), or None (uses default COCO image).

        Returns:
            torch.Tensor: Preprocessed input tensor [batch_size, 3, H, W].
        """
        stride = 32
        input_size = 640
        img_size = check_img_size(input_size, s=stride)

        # If no image provided, use default COCO image
        if image is None:
            cached_path = get_file(
                "http://images.cocodataset.org/val2017/000000397133.jpg"
            )
            img_src = np.asarray(Image.open(cached_path).convert("RGB"))
        else:
            # Convert image to numpy array (RGB format) if needed
            if isinstance(image, Image.Image):
                img_src = np.asarray(image.convert("RGB"))
            elif isinstance(image, str):
                # Assume it's a file path or URL
                if image.startswith(("http://", "https://")):
                    cached_path = get_file(image)
                    img_src = np.asarray(Image.open(cached_path).convert("RGB"))
                else:
                    # Local file path
                    img_src = np.asarray(Image.open(image).convert("RGB"))
            elif isinstance(image, torch.Tensor):
                # Convert tensor to numpy array
                img_src = image.cpu().numpy()
                if img_src.dtype != np.uint8:
                    img_src = (img_src * 255).astype(np.uint8)
                # Assume CHW format, convert to HWC
                if len(img_src.shape) == 3 and img_src.shape[0] == 3:
                    img_src = img_src.transpose((1, 2, 0))
                # Convert RGB to RGB (already RGB)
                if len(img_src.shape) == 3 and img_src.shape[2] == 3:
                    pass  # Already RGB
            elif isinstance(image, np.ndarray):
                img_src = image
                if len(img_src.shape) == 3 and img_src.shape[2] == 3:
                    # Assume BGR, convert to RGB
                    img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

        # Store original image for postprocessing
        self.img_src = img_src

        # Process image using letterbox
        image_processed = letterbox(img_src, img_size, stride=stride)[0]
        # Convert HWC to CHW
        image_processed = image_processed.transpose((2, 0, 1))
        image_processed = torch.from_numpy(np.ascontiguousarray(image_processed))
        image_processed = image_processed.float()  # uint8 to fp32
        image_processed /= 255  # 0 - 255 to 0.0 - 1.0

        # Add batch dimension
        input_batch = image_processed.unsqueeze(0)
        self.input_batch = input_batch

        # Handle batch size
        if batch_size > 1:
            input_batch = input_batch.repeat(batch_size, 1, 1, 1)

        # Apply dtype override if specified
        if dtype_override is not None:
            input_batch = input_batch.to(dtype_override)

        return input_batch

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs for the model.

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

    def output_postprocess(self, output):
        """Post-process model outputs.

        Args:
            output: Model output tensor.

        Returns:
            dict: Dictionary containing:
                - detections: List of detection results per sample
                - num_detections: Number of detections per sample
        """
        det = non_max_suppression(output.detach().float())

        coco_yaml_path = get_file("test_files/pytorch/yolo/coco.yaml")
        with open(coco_yaml_path, "r") as f:
            coco_yaml = yaml.safe_load(f)
        class_names = coco_yaml["names"]

        results = []
        if len(det) and self.input_batch is not None and self.img_src is not None:
            for sample in range(self.input_batch.shape[0]):
                det[sample][:, :4] = Inferer.rescale(
                    self.input_batch.shape[2:], det[sample][:, :4], self.img_src.shape
                ).round()

                sample_detections = []
                for *xyxy, conf, cls in reversed(det[sample]):
                    class_num = int(cls)
                    conf_value = conf.item()
                    coordinates = [int(x.item()) for x in xyxy]
                    label = class_names[class_num]

                    sample_detections.append(
                        {
                            "coordinates": coordinates,
                            "class": label,
                            "confidence": conf_value,
                        }
                    )

                results.append(
                    {
                        "sample_id": sample,
                        "detections": sample_detections,
                        "num_detections": len(sample_detections),
                    }
                )

        return {
            "detections": results,
            "num_samples": len(results),
        }
