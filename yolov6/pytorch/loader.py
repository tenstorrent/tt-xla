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
        self.img_src = img_src
        input_batch = img.unsqueeze(0)
        self.input_batch = input_batch
        # Replicate tensors for batch size
        batch_tensor = input_batch.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor

    def post_process(self, output):
        """Post-process the output of the YOLOv6 model.
        Args:
            output: The output of the YOLOv6 model.
        Returns:
            The decoded output.
        """

        det = non_max_suppression(output.detach().float())

        coco_yaml_path = get_file("test_files/pytorch/yolo/coco.yaml")
        with open(coco_yaml_path, "r") as f:
            coco_yaml = yaml.safe_load(f)
        class_names = coco_yaml["names"]

        if len(det):
            for sample in range(self.input_batch.shape[0]):
                print("Sample ID: ", sample)
                det[sample][:, :4] = Inferer.rescale(
                    self.input_batch.shape[2:], det[sample][:, :4], self.img_src.shape
                ).round()

                for *xyxy, conf, cls in reversed(det[sample]):
                    class_num = int(cls)  # Convert class index to integer
                    conf_value = conf.item()  # Get the confidence value
                    coordinates = [
                        int(x.item()) for x in xyxy
                    ]  # Convert tensor to list of integers

                    # Get the class label
                    label = class_names[class_num]

                    # Detections
                    print(
                        f"Coordinates: {coordinates}, Class: {label}, Confidence: {conf_value:.2f}"
                    )
                print("\n")
