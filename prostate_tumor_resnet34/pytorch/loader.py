# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Prostate Tumor ResNet34 model loader implementation for image classification
"""

from torchvision import models, transforms
from typing import Optional
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
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


class ModelVariant(StrEnum):
    """Available Prostate Tumor ResNet34 model variants."""

    TCGA_PRAD = "TCGA_PRAD"


class ModelLoader(ForgeModel):
    """Prostate Tumor ResNet34 model loader for histopathology image classification."""

    _VARIANTS = {
        ModelVariant.TCGA_PRAD: ModelConfig(
            pretrained_model_name="kaczmarj/prostate-tumor-resnet34.tcga-prad",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TCGA_PRAD

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Prostate Tumor ResNet34",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Prostate Tumor ResNet34 model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The ResNet34 model instance with 3 output classes.
        """
        model = models.resnet34(num_classes=3)

        weights_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="model.safetensors",
        )
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Batch size for the input tensor.

        Returns:
            torch.Tensor: Preprocessed input tensor of shape [batch, 3, 224, 224].
        """
        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.6462, 0.507, 0.8055],
                    std=[0.1381, 0.1674, 0.1358],
                ),
            ]
        )

        image = Image.new("RGB", (224, 224), (160, 130, 200))
        input_tensor = preprocess(image).unsqueeze(0)

        if batch_size > 1:
            input_tensor = input_tensor.expand(batch_size, -1, -1, -1).contiguous()

        if dtype_override is not None:
            input_tensor = input_tensor.to(dtype_override)

        return input_tensor
