# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ORMBG ONNX model loader implementation for image segmentation (background removal).
"""

from typing import Optional

import onnx
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
    """Available ORMBG ONNX model variants."""

    ORMBG = "ormbg"


class ModelLoader(ForgeModel):
    """ORMBG ONNX model loader implementation for background removal."""

    _VARIANTS = {
        ModelVariant.ORMBG: ModelConfig(
            pretrained_model_name="onnx-community/ormbg-ONNX",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ORMBG

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ORMBG",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the ORMBG ONNX model.

        Returns:
            onnx.ModelProto: The ONNX model instance.
        """
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="onnx/model.onnx",
        )
        model = onnx.load(local_path)

        return model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the ORMBG model.

        The model expects images of shape [1, 3, 1024, 1024] rescaled to [0, 1].

        Returns:
            torch.Tensor: Sample input tensor of shape [1, 3, 1024, 1024].
        """
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB")

        preprocess = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
            ]
        )
        input_tensor = preprocess(image)
        inputs = input_tensor.unsqueeze(0)

        return inputs
