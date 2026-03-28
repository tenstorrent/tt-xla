# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Scrypted YOLOv9 ReLU ONNX model loader implementation
"""

from typing import Optional

import numpy as np
import onnx
import torch
from datasets import load_dataset
from torchvision import transforms

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Scrypted YOLOv9 ReLU ONNX model variants."""

    T = "T"
    S = "S"
    M = "M"
    C = "C"


# Map variant to HuggingFace ONNX file path
_VARIANT_TO_ONNX_PATH = {
    ModelVariant.T: "onnx/scrypted_yolov9t_relu/scrypted_yolov9t_relu.onnx",
    ModelVariant.S: "onnx/scrypted_yolov9s_relu/scrypted_yolov9s_relu.onnx",
    ModelVariant.M: "onnx/scrypted_yolov9m_relu/scrypted_yolov9m_relu.onnx",
    ModelVariant.C: "onnx/scrypted_yolov9c_relu/scrypted_yolov9c_relu.onnx",
}


class ModelLoader(ForgeModel):
    """Scrypted YOLOv9 ReLU ONNX model loader implementation."""

    _VARIANTS = {
        ModelVariant.T: ModelConfig(pretrained_model_name="scrypted/plugin-models"),
        ModelVariant.S: ModelConfig(pretrained_model_name="scrypted/plugin-models"),
        ModelVariant.M: ModelConfig(pretrained_model_name="scrypted/plugin-models"),
        ModelVariant.C: ModelConfig(pretrained_model_name="scrypted/plugin-models"),
    }

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

        return ModelInfo(
            model="Scrypted_YOLOv9_ReLU",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the Scrypted YOLOv9 ReLU ONNX model.

        Returns:
            onnx.ModelProto: The ONNX model instance.
        """
        from huggingface_hub import hf_hub_download

        onnx_path = _VARIANT_TO_ONNX_PATH[self._variant]
        local_path = hf_hub_download(
            repo_id="scrypted/plugin-models",
            filename=onnx_path,
        )
        model = onnx.load(local_path)

        return model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the Scrypted YOLOv9 ReLU model.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB").resize((640, 640))
        m, s = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))
        preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=m, std=s)]
        )
        input_tensor = preprocess(image)
        inputs = input_tensor.unsqueeze(0)

        return inputs
