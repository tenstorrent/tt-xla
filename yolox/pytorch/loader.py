# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOX model loader implementation
"""

import torch
import os
from typing import Optional

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

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
from .src.utils import _forward_patch, _decode_outputs


class ModelVariant(StrEnum):
    """Available YOLOX model variants."""

    YOLOX_NANO = "yolox_nano"
    YOLOX_TINY = "yolox_tiny"
    YOLOX_S = "yolox_s"
    YOLOX_M = "yolox_m"
    YOLOX_L = "yolox_l"
    YOLOX_DARKNET = "yolox_darknet"
    YOLOX_X = "yolox_x"


class ModelLoader(ForgeModel):
    """YOLOX model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.YOLOX_NANO: ModelConfig(
            pretrained_model_name="yolox_nano",
        ),
        ModelVariant.YOLOX_TINY: ModelConfig(
            pretrained_model_name="yolox_tiny",
        ),
        ModelVariant.YOLOX_S: ModelConfig(
            pretrained_model_name="yolox_s",
        ),
        ModelVariant.YOLOX_M: ModelConfig(
            pretrained_model_name="yolox_m",
        ),
        ModelVariant.YOLOX_L: ModelConfig(
            pretrained_model_name="yolox_l",
        ),
        ModelVariant.YOLOX_DARKNET: ModelConfig(
            pretrained_model_name="yolox_darknet",
        ),
        ModelVariant.YOLOX_X: ModelConfig(
            pretrained_model_name="yolox_x",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.YOLOX_NANO

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
            model="yolox",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the YOLOX model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOX model instance.
        """
        from yolox.exp import get_exp  # Defer heavy import
        from yolox.models.yolo_head import YOLOXHead

        YOLOXHead.forward = _forward_patch
        YOLOXHead.decode_outputs = _decode_outputs

        # Get the model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        weight_url = f"https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{model_name}.pth"

        # Use the utility to download/cache the model weights
        weight_path = get_file(weight_url)

        # Handle special case for darknet variant
        if model_name == "yolox_darknet":
            exp_name = "yolov3"
        else:
            exp_name = model_name.replace("_", "-")

        # Load model
        exp = get_exp(exp_name=exp_name)
        model = exp.get_model()
        ckpt = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the YOLOX model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """
        # Deter imports so not required at model discovery time
        from datasets import load_dataset
        from yolox.data.data_augment import preproc as preprocess

        # Determine input shape based on model variant
        model_name = self._variant_config.pretrained_model_name
        if model_name in ["yolox_nano", "yolox_tiny"]:
            input_shape = (416, 416)
        else:
            input_shape = (640, 640)

        ds = load_dataset("mpnikhil/kitchen-classifier", split="train").with_format(
            "np"
        )  # to get the image as an numpy array
        img = ds[200]["image"]
        img_tensor, ratio = preprocess(img, input_shape)
        self.ratio = ratio
        self.input_shape = input_shape  # Store for post_processing
        img_tensor = torch.from_numpy(img_tensor)
        batch_tensor = img_tensor.unsqueeze(0)

        # Replicate tensors for batch size
        batch_tensor = batch_tensor.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor

    def post_processing(self, co_out):
        """Post-process the model outputs.

        Args:
            co_out: Compiled model outputs

        Returns:
            None: Prints the detection results
        """
        from .src.utils import print_detection_results

        print_detection_results(co_out, self.ratio, self.input_shape)
