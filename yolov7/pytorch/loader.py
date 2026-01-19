# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv7 model loader implementation
"""

from typing import Optional
import torch
import numpy as np
from PIL import Image

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

from third_party.tt_forge_models.yolov7.pytorch.src.model_utils import (
    check_img_size,
    attempt_load,
    letterbox,
    yolov7_postprocess,
)
from third_party.tt_forge_models.tools.utils import get_file


class ModelVariant(StrEnum):
    """Available YOLOv7 model variants (detection)."""

    YOLOV7 = "yolov7"
    YOLOV7X = "yolov7x"
    YOLOV7_W6 = "yolov7-w6"
    YOLOV7_E6 = "yolov7-e6"
    YOLOV7_D6 = "yolov7-d6"
    YOLOV7_E6E = "yolov7-e6e"


class ModelLoader(ForgeModel):
    """YOLOv7 model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.YOLOV7X: ModelConfig(
            pretrained_model_name="yolov7x",
        ),
        ModelVariant.YOLOV7: ModelConfig(
            pretrained_model_name="yolov7",
        ),
        ModelVariant.YOLOV7_W6: ModelConfig(
            pretrained_model_name="yolov7-w6",
        ),
        ModelVariant.YOLOV7_E6: ModelConfig(
            pretrained_model_name="yolov7-e6",
        ),
        ModelVariant.YOLOV7_D6: ModelConfig(
            pretrained_model_name="yolov7-d6",
        ),
        ModelVariant.YOLOV7_E6E: ModelConfig(
            pretrained_model_name="yolov7-e6e",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.YOLOV7

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

        if variant in [ModelVariant.YOLOV7]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="yolov7",
            variant=variant,
            group=group,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the YOLOv7 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                        If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOv7 model instance.
        """
        from urllib.parse import urljoin

        # Get model name from variant
        model_name = self._variant_config.pretrained_model_name

        weights_map = {
            "yolov7x": "yolov7x.pt",
            "yolov7": "yolov7.pt",
            "yolov7-w6": "yolov7-w6.pt",
            "yolov7-e6": "yolov7-e6.pt",
            "yolov7-d6": "yolov7-d6.pt",
            "yolov7-e6e": "yolov7-e6e.pt",
        }

        weight_name = weights_map.get(model_name, f"{model_name}.pt")
        base_url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/"
        remote_url = urljoin(base_url, weight_name)
        weight_path = get_file(remote_url)
        model = attempt_load(weight_path, map_location="cpu")
        model.eval()

        self.model = model
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(
        self,
        dtype_override=None,
    ):
        """Load and return sample inputs for the YOLOv6 model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """

        stride = 32
        img_size = 640
        img_size = check_img_size(img_size, s=stride)

        # Resolve image
        image_path = get_file("http://images.cocodataset.org/val2017/000000298251.jpg")

        img_src = np.asarray(Image.open(image_path).convert("RGB"))
        self.img_src = img_src
        img_lb, _, _ = letterbox(
            img_src,
            new_shape=img_size,
            auto=False,
            scaleFill=False,
            scaleup=False,
            stride=stride,
        )
        img_chw = img_lb.transpose((2, 0, 1))
        img = torch.from_numpy(np.ascontiguousarray(img_chw)).float() / 255.0
        input_batch = img.unsqueeze(0)
        self.input_batch = input_batch
        batch_tensor = input_batch.repeat_interleave(1, dim=0)
        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)
        return batch_tensor

    def post_process(self, co_out):
        """Post-process YOLOv7 model outputs to extract detection results.

        Args:
            co_out: Raw model output tensor from YOLOv7 forward pass.

        Returns:
            Post-processed detection results.
        """
        return yolov7_postprocess(co_out, self.img_src, self.input_batch)
