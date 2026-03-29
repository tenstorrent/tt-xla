# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OnnxTR CRNN MobileNet V3 Large ONNX model loader implementation for text recognition.
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
    """Available OnnxTR CRNN MobileNet V3 Large model variants."""

    CRNN_MOBILENET_V3_LARGE = "crnn-mobilenet-v3-large"


class ModelLoader(ForgeModel):
    """OnnxTR CRNN MobileNet V3 Large ONNX model loader implementation."""

    _VARIANTS = {
        ModelVariant.CRNN_MOBILENET_V3_LARGE: ModelConfig(
            pretrained_model_name="Felix92/onnxtr-crnn-mobilenet-v3-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CRNN_MOBILENET_V3_LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="onnxtr_crnn_mobilenet_v3_large",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the CRNN MobileNet V3 Large ONNX model.

        Returns:
            onnx.ModelProto: The ONNX model instance.
        """
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="model.onnx",
        )
        model = onnx.load(local_path)

        return model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the CRNN MobileNet V3 Large model.

        The model expects images of shape [1, 3, 32, 128] normalized with
        mean=[0.694, 0.695, 0.693] and std=[0.299, 0.296, 0.301].

        Returns:
            torch.Tensor: Sample input tensor of shape [1, 3, 32, 128].
        """
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB").resize((128, 32))

        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.694, 0.695, 0.693],
                    std=[0.299, 0.296, 0.301],
                ),
            ]
        )
        input_tensor = preprocess(image)
        inputs = input_tensor.unsqueeze(0)

        return inputs
